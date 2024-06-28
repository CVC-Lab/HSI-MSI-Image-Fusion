import numpy as np
import scipy.io as sio
import cv2
import spectral as spy
import pdb
from einops import rearrange
import torch
from .base_dataset import BaseSegmentationDataset
from .contrast_enhancement import contrast_enhancement

RGB = np.array([630.0, 532.0, 465.0])

def input_processing(img_path, gt_path):
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
    return img_sri, gt


# lets just use gamma correction
def adjust_gamma_hyperspectral(image, gamma=0.5):
    epsilon = 1e-8
    # Calculate the inverse gamma
    invGamma = 1.0 / gamma
    # Initialize an array to store the gamma-corrected image
    gamma_corrected_image = np.zeros_like(image)
    # Apply gamma correction to each spectral band
    for band in range(image.shape[2]):
        # Normalize the band to [0, 1]
        normalized_band = image[:, :, band] / (np.max(image[:, :, band]) + epsilon)
        # Apply gamma correction
        gamma_corrected_band = np.power(normalized_band, invGamma)
        # Scale back to original range
        gamma_corrected_image[:, :, band] = gamma_corrected_band * np.max(image[:, :, band])
    return gamma_corrected_image


class JasperRidgeDataset(BaseSegmentationDataset):
    ''' Simple dataset from subimage of a single HSI image'''
    def __init__(self,
                 single_img_path, single_gt_path,
                 start_band, end_band, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height,
                 channels=None, 
                 mode="train", 
                 transforms=None, 
                 split_ratio=0.8, seed=42, 
                 window_size=5, conductivity=0.95,
                 **kwargs):
        self.channels = channels
        self.start_band = start_band
        self.end_band = end_band
        img_sri, gt = input_processing(single_img_path, single_gt_path)
        img_sri = adjust_gamma_hyperspectral(img_sri, gamma=0.4)
        # img_sri = contrast_enhancement((img_sri*255).astype(np.uint8), 
        #                                     window_size=window_size, 
        #                                     conductivity=conductivity)/255
        img_rgb = self.get_rgb(img_sri)
        super().__init__(img_sri=img_sri, 
                         img_rgb=img_rgb,
                         gt=gt,
                         rgb_width=rgb_width,
                         rgb_height=rgb_height, hsi_width=hsi_width, 
                         hsi_height=hsi_height,
                         channels=channels, 
                         mode=mode, transforms=transforms, 
                         split_ratio=split_ratio, seed=seed, stride=1)
        
        
    def get_rgb(self, img_sri):
        num_bands = img_sri.shape[-1]
        # Infer RGB from SRI 
        dist_band = (self.end_band - self.start_band)/ (num_bands - 1)
        RGB_indices = np.array((RGB - self.start_band)/dist_band, dtype=int)
        img_rgb = spy.get_rgb(img_sri, (RGB_indices[0], RGB_indices[1], RGB_indices[2]))
        return img_rgb