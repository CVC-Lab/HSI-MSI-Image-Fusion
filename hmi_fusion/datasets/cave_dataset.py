from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
from PIL import Image
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

def para_setting(kernel_type,sf,sz,sigma):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


test_classes = ["balloons_ms", "cd_ms", "cloth_ms", "photo_and_face_ms", "thread_spools_ms"]


def load_data(dataset_dir, mode):
    data =  []
    dataset_dir = os.path.join(dataset_dir)
    classes = os.listdir(dataset_dir)
    class2id = {c: idx for idx, c in enumerate(classes)}
    if mode == "test":
        classes = test_classes
    else:
        classes = [c for c in classes if c not in test_classes]
    
    for c in os.listdir(dataset_dir):
        if c not in classes: continue
        
        c_path = os.path.join(dataset_dir, c, c)
        if not os.path.isdir(c_path): continue
        ms_path = []
        rgb_path = None
        for im in os.listdir(c_path):
            if "_ms_" in im:
                ms_path.append(os.path.join(c_path, im))
            elif im.endswith(".bmp"):
                rgb_path = os.path.join(c_path, im)
        
        ms_img = load_ms_img(ms_path=ms_path)
        rgb_img = np.array(Image.open(rgb_path))
        data.append((c, ms_img, rgb_img))
    return classes, class2id, data
        

class CAVEDataset(Dataset):
    def __init__(self, dataset_dir, mode="train", sf=8, transform=ToTensor()) -> None:
        super().__init__()
        self.transform = transform
        self.classes, self.class2id, self.data = load_data(dataset_dir, mode)
        self.sizeI = 512
        self.factor = sf # scaling factor
        self.mode = mode

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
        # need to convert to Tensor
        # c, ms_img, rgb_img = self.data[idx]
        c, HR_HSI, HR_MSI = self.data[idx]
        sz = [self.sizeI, self.sizeI]
        sigma = 2.0
        fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma)
        fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)),2)
        fft_BT = torch.cat((torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2)

        px      = random.randint(0, 512-self.sizeI)
        py      = random.randint(0, 512-self.sizeI)
        hr_hsi  = HR_HSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
        hr_msi  = HR_MSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]

        if self.mode == "train":
            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hr_hsi  =  np.rot90(hr_hsi)
                hr_msi  =  np.rot90(hr_msi)

            # Random vertical Flip
            for j in range(vFlip):
                hr_hsi = hr_hsi[:, ::-1, :].copy()
                hr_msi = hr_msi[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                hr_hsi = hr_hsi[::-1, :, :].copy()
                hr_msi = hr_msi[::-1, :, :].copy()

        hr_hsi = torch.FloatTensor(hr_hsi.copy()).permute(2,0,1).unsqueeze(0)
        hr_msi = torch.FloatTensor(hr_msi.copy()).permute(2,0,1).unsqueeze(0)
        lr_hsi = self.H_z(hr_hsi, self.factor, fft_B)
        lr_hsi = torch.FloatTensor(lr_hsi)

        hr_hsi = hr_hsi.squeeze(0)
        hr_msi = hr_msi.squeeze(0)
        lr_hsi = lr_hsi.squeeze(0)
        # print(f'lr_hsi.shape: {lr_hsi.shape}, hr_hsi.shape: {hr_hsi.shape}, hr_msi.shape: {hr_msi.shape}')
        # [31, 64, 64], [31, 512, 512], [3, 512, 512]
        return lr_hsi, hr_msi, hr_hsi # Yh, Ym, X

# dataset = CAVEDataset("./data/CAVE", mode="train")
# pdb.set_trace()
