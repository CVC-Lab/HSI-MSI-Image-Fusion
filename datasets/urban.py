import numpy as np
import scipy.io as sio
import cv2
import spectral as spy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pdb
from einops import rearrange
import torch
from .utils import psf2otf
from pathlib import Path

"""
URBAN dataset:
-------------
Urban is one of the most widely used hyperspectral data used in the 
hyperspectral unmixing study. There are 307x307 pixels, each of which 
corresponds to a 2x2 m2 area. In this image, there are 210 wavelengths 
ranging from 400 nm to 2500 nm, resulting in a spectral resolution of 10 nm. 
After the channels 1-4, 76, 87, 101-111, 136-153 and 198-210 are removed 
"""
wavelengths = np.linspace(400, 2500, 210)
wavelengths = np.delete(wavelengths, 
                        [0, 1, 2, 3, 75, 86] + list(range(100,110)) + 
                        list(range(135, 152)) + list(range(197, 209)))
# Desired RGB wavelengths
RGB_wavelengths = np.array([630.0, 532.0, 465.0])

# Find the closest bands for the desired RGB wavelengths
def find_closest_bands(wavelengths, target_wavelengths):
    closest_bands = []
    for target in target_wavelengths:
        index = (np.abs(wavelengths - target)).argmin()
        closest_bands.append(index)
    return closest_bands


def para_setting(kernel_type, sf, sz, sigma):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), 
                          (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


# Function to process the ground truth data
def process_ground_truth_data(tmp):
    nRow = tmp['nRow'][0, 0]
    nCol = tmp['nCol'][0, 0]
    nEnd = tmp['nEnd'][0, 0]
    # Reshape the A array to (nEnd, nRow, nCol)
    A = tmp['A'].reshape((nEnd, nRow, nCol))
    return A

# Function to process the hyperspectral data
def process_hyperspectral_data(img):
    nRow = img['nRow'][0, 0]
    nCol = img['nCol'][0, 0]
    nBand = img['Y'].shape[0]
    maxValue = img['maxValue'][0, 0]
    
    # Reshape and normalize the data
    hyperspectral_image = img['Y'].reshape((nBand, nRow, nCol)).transpose(1, 2, 0)
    hyperspectral_image = hyperspectral_image / maxValue
    closest_bands = find_closest_bands(wavelengths, RGB_wavelengths)
    # Extract and normalize the RGB bands
    R_band = hyperspectral_image[:, :, closest_bands[0]]
    G_band = hyperspectral_image[:, :, closest_bands[1]]
    B_band = hyperspectral_image[:, :, closest_bands[2]]
    # Stack bands to create the RGB image
    RGB_image = np.stack((R_band, G_band, B_band), axis=-1)
    return hyperspectral_image, RGB_image


def input_processing(img_path, gt_path, start_band=None, end_band=None):
    '''Processing mat files input images to output a tuple of (full SRI, RGB, and gt mask)'''
    # Get ground truth label
    tmp = sio.loadmat(gt_path)
    gt = process_ground_truth_data(tmp)
    # Get full SRI
    img = sio.loadmat(img_path) # (162, 94249)
    img_sri, img_rgb = process_hyperspectral_data(img)
    gt = rearrange(gt, "C H W -> H W C")

    return img_sri, img_rgb, gt


class UrbanDataset(Dataset):
    ''' Simple dataset from subimage of a single HSI image'''
    def __init__(self, data_dir, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height, 
                 mode="train", transforms=None, split_ratio=0.8, seed=42):
        
        data_dir = Path(data_dir)
        single_img_path = data_dir / "Urban_R162.mat"
        single_gt_path = data_dir / "groundTruth_Urban_end6/end6_groundTruth.mat"
        processed_input = input_processing(single_img_path, single_gt_path)
        self.img_sri, self.img_rgb = processed_input[0], processed_input[1]
        self.gt = processed_input[2]
        self.num_classes = self.gt.shape[-1]
        # if channels == None:
        #     pass
        #     # self.img_sri = get_most_informative_img_sri(self.img_sri, self.gt, self.num_classes)
        # else:
        #     self.img_sri = self.img_sri[:, :, channels]
        self.width, self.height = self.gt.shape[0], self.gt.shape[1]
        self.rgb_width, self.rgb_height = rgb_width, rgb_height
        self.hsi_width, self.hsi_height = hsi_width, hsi_height
        self.factor = self.rgb_width // self.hsi_width
        self.sizeI = self.width
        self.sigma = 2
        # Max indices along width/height dimension for subimage extraction.
        self.max_width_index = self.width - self.rgb_width + 1
        self.max_height_index = self.height - self.rgb_height + 1
        stride = 8 # important for controlling size of dataset
        indices = list(range(0, self.max_width_index * self.max_height_index, stride))
        train_indices, test_indices = train_test_split(
            indices, test_size=(1 - split_ratio), random_state=seed
        )
        self.mode = mode
        if self.mode == 'train':
            self.indices = train_indices
        else:
            self.indices = test_indices
        self.transforms = transforms

    def __len__(self):
        return len(self.indices)
    
    def H_z(self, z: torch.Tensor, factor: float, fft_B: torch.Tensor):
        # f = torch.fft.rfft(z, 2, onesided=False)
        f = torch.fft.fft2(z) # [1, 31, 512, 512]
        f = torch.view_as_real(f) #[1, 31, 512, 512, 2]
        
        # -------------------complex myltiply-----------------#
        if len(z.shape) == 3:
            ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
            M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
                           (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
            # Hz = torch.fft.irfft(M, 2, onesided=False)
            Hz = torch.fft.ifft2(torch.view_as_complex(M))
            x = Hz[:, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
        elif len(z.shape) == 4:
            bs, ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
            M = torch.cat(
                ((f[:, :, :, :, 0] * fft_B[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_B[:, :, :, :, 1]).unsqueeze(4),
                 (f[:, :, :, :, 0] * fft_B[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_B[:, :, :, :, 0]).unsqueeze(4)), 4)
            # Hz = torch.irfft(M, 2, onesided=False)
            Hz = torch.fft.ifft2(torch.view_as_complex(M))
            x = Hz[:, :, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
       
        return torch.view_as_real(x)[..., 0]
    
    def downsample(self, sub_sri):
        sub_sri = torch.FloatTensor(sub_sri).permute(2, 1, 0)
        sz = [sub_sri.shape[1], sub_sri.shape[2]]
        fft_B, _ = para_setting('gaussian_blur', self.factor, sz, self.sigma)
        self.fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), 
                                torch.Tensor(np.imag(fft_B)).unsqueeze(2)),2)
        sub_hsi = self.H_z(sub_sri, self.factor, self.fft_B).permute(2, 1, 0).numpy()
        return sub_hsi
    
    def __getitem__(self, idx):
        # idx is in range [0, len(indices)] but 
        # self.indices stores actual indices of bigger ds
        
        idx = self.indices[idx] 
        width_index = idx // self.max_height_index
        height_index = idx % self.max_height_index

        # Get RGB instance and ground-truth mask
        sub_rgb = self.img_rgb[width_index:(width_index + self.rgb_width), 
                               height_index:(height_index + self.rgb_height), :]
        sub_gt = self.gt[width_index:(width_index + self.rgb_width), 
                               height_index:(height_index + self.rgb_height), :]
        sub_sri = self.img_sri[width_index:(width_index + self.rgb_width), 
                               height_index:(height_index + self.rgb_height), :]
        
        sub_hsi = self.downsample(sub_sri)
        
        # Original shapes: (H, W, CH) and (H, W, 3), (H, W, N).
        # sub_hsi = sub_hsi[:, :, self.channels]
        if self.transforms:
            sub_hsi, sub_rgb, sub_gt = self.transforms(sub_hsi, sub_rgb, sub_gt)
            
        # After conversion shapes: (CH, H, W), (3, H, W), (N, H, W).
        else:
            sub_hsi = np.moveaxis(sub_hsi, 2, 0)
            sub_rgb = np.moveaxis(sub_rgb, 2, 0)
            sub_gt = np.moveaxis(sub_gt, 2, 0)
        
        return sub_hsi, sub_rgb, sub_gt


if __name__ == '__main__':
    data_dir = "/mnt/data/shubham/hsi_msi/urban"
    ds = UrbanDataset(data_dir=data_dir, rgb_height=64, rgb_width=64, hsi_height=32, hsi_width=32)
    hsi, rgb, gt = ds[0]
    print(hsi.shape, rgb.shape, gt.shape)