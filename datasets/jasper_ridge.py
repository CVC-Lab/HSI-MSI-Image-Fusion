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

RGB = np.array([630.0, 532.0, 465.0])

def para_setting(kernel_type, sf, sz, sigma):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


def input_processing(img_path, gt_path, start_band, end_band):
    '''Processing mat files input images to output a tuple of (full SRI, RGB, and gt mask)'''
    # Get ground truth label
    tmp = sio.loadmat(gt_path)
    gt = tmp['data']
    width, height = gt.shape[0], gt.shape[1]
    # Get full SRI
    tmp = sio.loadmat(img_path)
    max_value = tmp['maxValue'][0][0]
    img_sri = tmp['Y']/max_value
    img_sri = img_sri.swapaxes(0, 1).reshape(height, width, -1).swapaxes(0, 1)   
    num_bands = img_sri.shape[-1]

    # Infer RGB from SRI 
    dist_band = (end_band - start_band)/ (num_bands - 1)
    RGB_indices = np.array((RGB - start_band)/dist_band, dtype=int)
    img_rgb = spy.get_rgb(img_sri, (RGB_indices[0], RGB_indices[1], RGB_indices[2]))
    
    return img_sri, img_rgb, gt


class JasperRidgeDataset(Dataset):
    ''' Simple dataset from subimage of a single HSI image'''
    def __init__(self, channels,
                 single_img_path, single_gt_path,
                 start_band, end_band, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height, 
                 mode="train", transforms=None, split_ratio=0.8, seed=42):
        self.channels = channels
        processed_input = input_processing(single_img_path, single_gt_path,
                                           start_band, end_band)
        self.img_sri, self.img_rgb = processed_input[0], processed_input[1]
        self.gt = processed_input[2]
        self.num_classes = self.gt.shape[-1]
        if channels == None:
            pass
            # self.img_sri = get_most_informative_img_sri(self.img_sri, self.gt, self.num_classes)
        else:
            self.img_sri = self.img_sri[:, :, channels]
        self.width, self.height = self.gt.shape[0], self.gt.shape[1]
        self.rgb_width, self.rgb_height = rgb_width, rgb_height
        self.hsi_width, self.hsi_height = hsi_width, hsi_height
        self.factor = self.rgb_width // self.hsi_width
        self.sizeI = 100
        self.sigma = 2
        # Max indices along width/height dimension for subimage extraction.
        self.max_width_index = self.width - self.rgb_width + 1
        self.max_height_index = self.height - self.rgb_height + 1
        
        indices = list(range(self.max_width_index * self.max_height_index))
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
            # sub_gt = rearrange(sub_gt, "H W C -> C H W")
        # After conversion shapes: (CH, H, W), (3, H, W), (N, H, W).
        else:
            sub_hsi = np.moveaxis(sub_hsi, 2, 0)
            sub_rgb = np.moveaxis(sub_rgb, 2, 0)
            sub_gt = np.moveaxis(sub_gt, 2, 0)
        
        return sub_hsi, sub_rgb, sub_gt
