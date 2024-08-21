import numpy as np
import scipy.io as sio
import cv2
import spectral as spy
from torch.utils.data import Dataset
import pdb
from einops import rearrange
import torch
from .base_dataset import BaseSegmentationDataset, adjust_gamma_hyperspectral
from .contrast_enhancement import contrast_enhancement
from train_utils.motioncode_selection import get_top_channels
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
    
    return hyperspectral_image


def input_processing(img_path, gt_path, start_band=None, end_band=None):
    '''Processing mat files input images to output a tuple of (full SRI, RGB, and gt mask)'''
    # Get ground truth label
    tmp = sio.loadmat(gt_path)
    gt = process_ground_truth_data(tmp)
    # Get full SRI
    img = sio.loadmat(img_path) # (162, 94249)
    img_sri = process_hyperspectral_data(img)
    gt = rearrange(gt, "C H W -> H W C")
    return img_sri, gt


class UrbanDataset(BaseSegmentationDataset):
    ''' Simple dataset from subimage of a single HSI image'''
    def __init__(self, data_dir, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height,
                 top_k, 
                 mode="train", transforms=None, split_ratio=0.8, seed=42,
                 channels=None,
                 window_size=5, conductivity=0.95, 
                 gamma=0.4, contrast_enhance=True,
                 **kwargs):
        data_dir = Path(data_dir)
        single_img_path = data_dir / "Urban_R162.mat"
        single_gt_path = data_dir / "groundTruth_Urban_end6/end6_groundTruth.mat"
        self.colors = ['red', 'green', 'purple', 'orange', 'gray', 'brown']
        self.label_names = ['Asphalt', 'Grass', 'Tree', 'Roof', 'Metal','Dirt']
        self.top_k = top_k
        num_classes =len(self.label_names)
        img_sri, gt = input_processing(single_img_path, single_gt_path)
        if channels != 'all':
            self.channels = get_top_channels(num_motion=num_classes,
                                             num_channels=img_sri.shape[-1], 
                                        top_k=self.top_k,
                                        dataset_name='urban')
        else:
            self.channels = list(range(0, img_sri.shape[-1]))
        
        img_sri = adjust_gamma_hyperspectral(img_sri, gamma=gamma)
        if contrast_enhance:
            img_sri = contrast_enhancement((img_sri*255).astype(np.uint8), 
                                                window_size=window_size, 
                                                conductivity=conductivity)/255
        img_rgb = self.get_rgb(img_sri)
        
        
        super().__init__(img_sri=img_sri, 
                         img_rgb=img_rgb,
                         gt=gt,
                         rgb_width=rgb_width,
                         rgb_height=rgb_height, hsi_width=hsi_width, 
                         hsi_height=hsi_height,
                         channels=self.channels, 
                         mode=mode, transforms=transforms, 
                         split_ratio=split_ratio, seed=seed, stride=8)
    
    def get_rgb(self, img_sri):
        closest_bands = find_closest_bands(wavelengths, RGB_wavelengths)
        # Extract and normalize the RGB bands
        R_band = img_sri[:, :, closest_bands[0]]
        G_band = img_sri[:, :, closest_bands[1]]
        B_band = img_sri[:, :, closest_bands[2]]
        # Stack bands to create the RGB image
        RGB_image = np.stack((R_band, G_band, B_band), axis=-1)
        return RGB_image


class MotionCodeUrban(UrbanDataset):
    
    def __init__(self, data_dir,
                 rgb_width, rgb_height,
                 hsi_width, hsi_height,
                 top_k=24, 
                 channels=None, 
                 mode="train", 
                 transforms=None, 
                 split_ratio=0.8, seed=42, 
                 window_size=5, conductivity=0.95,
                 gamma=0.4, contrast_enhance=True,
                 **kwargs):
        ## TODO: call init of JasperRidgeDataset
        self.top_k = top_k
        super().__init__(data_dir=data_dir, 
                         rgb_width=rgb_width, 
                         rgb_height=rgb_height, 
                         hsi_width=hsi_width, 
                         hsi_height=hsi_height, top_k=top_k,
                         channels=channels, mode=mode, 
                         transforms=transforms, 
                         split_ratio=split_ratio, seed=seed, 
                         window_size=window_size, 
                         conductivity=conductivity, 
                         gamma=gamma, 
                         contrast_enhance=contrast_enhance)
        self.Y_train, self.Y_all = None, None
        self.labels_train, self.labels_all = None, None
        self.img_hsi = None
        self.build_pixel_wise_dataset(size_each_class=1000)
        if mode == 'train':
            self.Y, self.labels = self.Y_train, self.labels_train
        else:
            self.Y, self.labels = self.Y_all, self.labels_all
        
        
    def __len__(self,):
        return self.Y.shape[0]
        
    def __getitem__(self, idx):
        return self.Y[idx], self.Y[idx], self.labels[idx]
    
    def __repr__(self):
        total_pixels = self.labels_all.shape[0]
        data_percentage = (self.Y.shape[0]/ total_pixels) * 100
        return f"dataset contains {data_percentage}% overall data"


if __name__ == '__main__':
    data_dir = "/mnt/data/shubham/hsi_msi/urban"
    ds = UrbanDataset(data_dir=data_dir, rgb_height=64, rgb_width=64, hsi_height=32, hsi_width=32)
    hsi, rgb, gt = ds[0]
    print(hsi.shape, rgb.shape, gt.shape)