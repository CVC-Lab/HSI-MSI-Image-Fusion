import numpy as np
import scipy.io as sio
import cv2
import spectral as spy
from torch.utils.data import Dataset


RGB = np.array([630.0, 532.0, 465.0])


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


class SingleImageDataset(Dataset):
    def __init__(self, channels,
                 single_img_path, single_gt_path,
                 start_band, end_band, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height):
        self.channels = channels
        processed_input = input_processing(single_img_path, single_gt_path,
                                           start_band, end_band)
        self.img_sri, self.img_rgb = processed_input[0], processed_input[1]
        self.gt = processed_input[2]
        self.width, self.height = self.gt.shape[0], self.gt.shape[1]
        self.rgb_width, self.rgb_height = rgb_width, rgb_height
        self.hsi_width, self.hsi_height = hsi_width, hsi_height

        # Max indices along width/height dimension for subimage extraction.
        self.max_width_index = self.width - self.rgb_width + 1
        self.max_height_index = self.height - self.rgb_height + 1

    def __len__(self):
        return self.max_width_index * self.max_height_index
    
    def __getitem__(self, idx):
        width_index = idx // self.max_height_index
        height_index = idx % self.max_height_index

        # Get RGB instance and ground-truth mask
        sub_rgb = self.img_rgb[width_index:(width_index + self.rgb_width), 
                               height_index:(height_index + self.rgb_height), :]
        sub_gt = self.gt[width_index:(width_index + self.rgb_width), 
                               height_index:(height_index + self.rgb_height), :]
        sub_sri = self.img_sri[width_index:(width_index + self.rgb_width), 
                               height_index:(height_index + self.rgb_height), :]
        
        # Down sample to get desired HSI instance.
        sub_hsi = cv2.pyrDown(sub_sri, dstsize=(self.hsi_width, self.hsi_height))

        # Original shapes: (H, W, CH) and (H, W, 3), (H, W, N).
        sub_hsi = sub_hsi[:, :, self.channels]

        # After conversion shapes: (CH, H, W), (3, H, W), (N, H, W).
        sub_hsi = np.moveaxis(sub_hsi, 2, 0)
        sub_rgb = np.moveaxis(sub_rgb, 2, 0)
        sub_gt = np.moveaxis(sub_gt, 2, 0)
        
        return sub_hsi, sub_rgb, sub_gt
