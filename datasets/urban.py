import numpy as np
import scipy.io as sio
import cv2
import spectral as spy
from torch.utils.data import Dataset
import pdb
from einops import rearrange
import torch
from .base_dataset import BaseSegmentationDataset
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


class UrbanDataset(BaseSegmentationDataset):
    ''' Simple dataset from subimage of a single HSI image'''
    def __init__(self, data_dir, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height, 
                 mode="train", transforms=None, split_ratio=0.8, seed=42):
        data_dir = Path(data_dir)
        single_img_path = data_dir / "Urban_R162.mat"
        single_gt_path = data_dir / "groundTruth_Urban_end6/end6_groundTruth.mat"
        processed_input = input_processing(single_img_path, single_gt_path)
        super().__init__(img_sri=processed_input[0], 
                         img_rgb=processed_input[1],
                         gt=processed_input[2],
                         rgb_width=rgb_width,
                         rgb_height=rgb_height, hsi_width=hsi_width, 
                         hsi_height=hsi_height,
                         channels=None, 
                         mode=mode, transforms=transforms, 
                         split_ratio=split_ratio, seed=seed, stride=8)


if __name__ == '__main__':
    data_dir = "/mnt/data/shubham/hsi_msi/urban"
    ds = UrbanDataset(data_dir=data_dir, rgb_height=64, rgb_width=64, hsi_height=32, hsi_width=32)
    hsi, rgb, gt = ds[0]
    print(hsi.shape, rgb.shape, gt.shape)