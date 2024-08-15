import numpy as np
import scipy.io as sio
import cv2
import spectral as spy
import pdb
from einops import rearrange
import torch
from .base_dataset import BaseSegmentationDataset, adjust_gamma_hyperspectral
from .contrast_enhancement import contrast_enhancement
from motion_code.data_processing import load_data, process_data_for_motion_codes
from train_utils.motioncode_selection import get_top_channels
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





class JasperRidgeDataset(BaseSegmentationDataset):
    ''' Simple dataset from subimage of a single HSI image'''
    def __init__(self,
                 single_img_path, single_gt_path,
                 start_band, end_band, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height,
                 top_k,
                 channels=None, 
                 mode="train", 
                 transforms=None, 
                 split_ratio=0.8, seed=42, 
                 window_size=5, conductivity=0.95,
                 gamma=0.4, contrast_enhance=True,
                 **kwargs):
        
        self.colors = ['purple', 'brown', 'blue', 'green']
        self.label_names = ['Road', 'Soil', 'Water', 'Tree']
        num_classes =len(self.label_names)
        self.top_k = top_k
        self.channels = get_top_channels(num_motion=num_classes, 
                                    top_k=self.top_k,
                                    dataset_name='jasper_ridge')
        
        self.start_band = start_band
        self.end_band = end_band
        img_sri, gt = input_processing(single_img_path, single_gt_path)
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
                         split_ratio=split_ratio, seed=seed, stride=1)
        
        
    def get_rgb(self, img_sri):
        num_bands = img_sri.shape[-1]
        # Infer RGB from SRI 
        dist_band = (self.end_band - self.start_band)/ (num_bands - 1)
        RGB_indices = np.array((RGB - self.start_band)/dist_band, dtype=int)
        img_rgb = spy.get_rgb(img_sri, (RGB_indices[0], RGB_indices[1], RGB_indices[2]))
        return img_rgb
    
    
class MotionCodeJasperRidge(JasperRidgeDataset):
    
    def __init__(self, single_img_path, single_gt_path,
                 start_band, end_band, 
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
        super().__init__(single_img_path, 
                         single_gt_path, 
                         start_band, 
                         end_band, 
                         rgb_width, 
                         rgb_height, hsi_width, hsi_height, top_k,
                         channels, mode, transforms, 
                         split_ratio, seed, window_size, 
                         conductivity, 
                         gamma, 
                         contrast_enhance)
        self.Y_train, self.Y_all = None, None
        self.labels_train, self.labels_all = None, None
        self.img_hsi = None
        
        self.build_pixel_wise_dataset()
        if mode == 'train':
            self.Y, self.labels = self.Y_train, self.labels_train
        else:
            self.Y, self.labels = self.Y_all, self.labels_all
        
    def get_pixel_coords(self, x):
        H, W, C = x.shape
        # Step 1: Generate the pixel coordinates
        # Generate a grid of coordinates
        y_coords, x_coords = torch.meshgrid(torch.arange(H), 
                                        torch.arange(W), indexing='ij')
        # Step 2: Flatten the spatial dimensions
        # Reshape x to have shape (C, H * W)
        x_flattened = x.reshape(-1, C)  # Flatten H and W
        x_coords_flattened = x_coords.flatten() # Flatten coordinates
        y_coords_flattened = y_coords.flatten()
        # Step 3: Concatenate coordinates and values
        # Stack coordinates to form (2, H * W)
        pixel_coordinates = torch.stack([x_coords_flattened, y_coords_flattened], dim=1)
        # Stack the coordinates and pixel values along the third dimension
        # Resulting shape: (H * W, 2 + C)
        result = torch.cat([pixel_coordinates, torch.from_numpy(x_flattened)], dim=1)
        return result
        
    
    def build_pixel_wise_dataset(self,):
        size_each_class = 50
        num_classes = len(self.label_names)
        self.img_hsi = self.downsample(self.img_sri)
        img_rgb, gt = self.img_rgb, self.gt
        gt = self.downsample(gt)
        img_hsi_reshaped = self.get_pixel_coords(self.img_hsi)
        gt_reshaped = gt.reshape(-1, gt.shape[-1])      
        indices = None
        all_labels = np.argmax(gt_reshaped, axis=1)
        for c in range(num_classes):
            indices_in_class = np.where(all_labels == c)[0]
            current_choices = np.random.choice(indices_in_class, size=size_each_class)
            if indices is None:
                indices = current_choices
            else:
                indices = np.append(indices, current_choices)
        num_series = indices.shape[0]
        all_num_series = img_hsi_reshaped.shape[0]
        self.Y_train = img_hsi_reshaped[indices, :].reshape(num_series, -1)
        self.Y_all = img_hsi_reshaped.reshape(all_num_series, -1)
        self.labels_all = gt_reshaped
        self.labels_train = gt_reshaped[indices, :]
        
    def __len__(self,):
        return self.Y.shape[0]
        
    def __getitem__(self, idx):
        return self.Y[idx], self.Y[idx], self.labels[idx]
    
    def __repr__(self):
        total_pixels = self.labels_all.shape[0]
        data_percentage = (self.Y.shape[0]/ total_pixels) * 100
        return f"dataset contains {data_percentage}% overall data"
        
        
        
