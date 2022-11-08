
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from scipy.ndimage.interpolation import rotate
import torch
from PIL import Image
import scipy.io as sio
import numpy as np
from .utils import psf2otf
import random
import cv2
import pdb
import os


def load_ms_img(ms_path):
    ms_img = []
    for p in ms_path:
        im = Image.open(p)
        ms_img.append(np.array(im))
    return np.moveaxis(np.array(ms_img), 0, -1)

def para_setting(kernel_type, sf, sz, sigma):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT

def load_data(dataset_dir, mode="train", train_split=0.6):
    data =  []
    folders = []
    # we have 27 images taken in under artificial or mixed illumination
    for folder in os.listdir(dataset_dir):
        if folder.startswith('.'): continue
        if folder == "matfiles": continue
        folder = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder): continue
        folders.append(sorted([os.path.join(folder, file)
                        for file in os.listdir(folder) if file.endswith(".mat")] ))

    # 60% for train - take 30 images from folder 1 and 16 images from folder 2
    for folder in folders:
        total = len(folder)
        if mode == "train":
            data = data + folder[:int(0.6*total)]
        elif mode == "test":
            data = data + folder[int(0.6*total):]
    
    loaded_data = []
    # mat = sio.loadmat(data[0])
    # loaded_data.append([mat['lbl'], mat['ref']])
    for file in data:
        mat = sio.loadmat(file)
        loaded_data.append([mat['lbl'], mat['ref'], 
        os.path.splitext(os.path.basename(file))[0]])

    return loaded_data
        

class HarvardDataset(Dataset):
    def __init__(self, dataset_dir, mode="train", sf=8) -> None:
        super().__init__()
        self.data = load_data(dataset_dir, mode)
        # img size - 1040, 1392
        self.sz = self.data[0][0].shape
        self.factor = sf # scaling factor

    def H_z(self, z, factor, fft_B):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        HR_HSI, HR_MSI, name = self.data[idx]
        size_r = HR_MSI.shape[0]
        HR_HSI = HR_HSI[:size_r, :size_r]
        HR_MSI = HR_MSI[:size_r, :size_r, :]

        # sz = [self.sizeI, self.sizeI]
        # print(HR_HSI.shape)
        sz = self.sz
        sigma = 2.0
        fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma)
        fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)),2)
        fft_BT = torch.cat((torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2)

        px      = random.randint(0, self.sz[0]-512)
        py      = random.randint(0, self.sz[1]-512)
        # print("hrhsi before:", HR_HSI.shape)
        hr_hsi = HR_HSI[..., None]
        hr_msi = HR_MSI

        # print("hrhsi after:", hr_hsi.shape)
        hr_hsi = torch.FloatTensor(hr_hsi.copy()).permute(2,0,1).unsqueeze(0)
        hr_msi = torch.FloatTensor(hr_msi.copy()).permute(2,0,1).unsqueeze(0)
        lr_hsi = self.H_z(hr_hsi, self.factor, fft_B)
        lr_hsi = torch.FloatTensor(lr_hsi)

        hr_hsi = hr_hsi.squeeze(0)
        hr_msi = hr_msi.squeeze(0)
        lr_hsi = lr_hsi.squeeze(0)
        # print(f'lr_hsi.shape: {lr_hsi.shape}, hr_hsi.shape: {hr_hsi.shape}, hr_msi.shape: {hr_msi.shape}')
        # lr_hsi.shape: torch.Size([1, 130, 174]), hr_hsi.shape: torch.Size([1, 1040, 1392]), hr_msi.shape: torch.Size([31, 1040, 1392])
        return name, lr_hsi, hr_msi, hr_hsi # Yh, Ym, X



# load_data("./datasets/data/Harvard")

# dataset = HarvardDataset("./data/Harvard", mode="test")
# pdb.set_trace()