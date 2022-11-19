
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
        # im = Image.open(p)
        im = cv2.imread(p, -1)
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

def load_data(dataset_dir, mode="train", sf=8, train_split=0.6):
    data =  []
    HSdir = os.path.join(dataset_dir, "HS")
    MSdir = os.path.join(dataset_dir, "MS")
    GTdir = os.path.join(dataset_dir, "GT")
    
    for im in os.listdir(GTdir):
        if im.endswith(".mat"):
            data.append(im)

    images = list(os.listdir(GTdir))
    if mode == "train":
        data + images[:int(0.6*len(images))]
    elif mode == "test":
        data + images[int(0.6*len(images)):]
    
    loaded_data = []
    for file in data:
        HS = sio.loadmat(os.path.join(HSdir, str(sf), file))
        MS = sio.loadmat(os.path.join(MSdir, file))
        GT = sio.loadmat(os.path.join(GTdir, file))
        loaded_data.append([HS['hsi'], MS['msi'], GT['ref'], 
        os.path.splitext(file)[0]])

    return loaded_data
        

class HarvardDataset(Dataset):
    def __init__(self, dataset_dir, mode="train", sf=8) -> None:
        super().__init__()
        self.data = load_data(dataset_dir, sf=sf, mode=mode)
        # img size - 1040, 1392
        self.sz = [1024, 1024]
        self.factor = sf # scaling factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        HSI, MSI, GT, name = self.data[idx]
        HSI = torch.FloatTensor(HSI).permute(2, 1, 0)
        MSI = torch.FloatTensor(MSI).permute(2, 1, 0)
        GT = torch.FloatTensor(GT).permute(2, 1, 0)

        HSI = HSI/HSI.max()
        MSI = MSI/MSI.max()
        GT = GT/GT.max()
        # print(f'lr_hsi.shape: {lr_hsi.shape}, hr_hsi.shape: {hr_hsi.shape}, hr_msi.shape: {hr_msi.shape}')
        # lr_hsi.shape: torch.Size([1, 130, 174]), hr_hsi.shape: torch.Size([1, 1040, 1392]), hr_msi.shape: torch.Size([31, 1040, 1392])
        # lr_hsi, hr_msi, hr_hsi
        return HSI, MSI, GT # Yh, Ym, X

        # return c, x_k, lr_hsi, hr_msi, hr_hsi, torch.T, yiq_downsampled, Zd, idx



# load_data("./datasets/data/Harvard")

# dataset = HarvardDataset("./datasets/data/Harvard", mode="test")
# pdb.set_trace()