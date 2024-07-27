import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from skimage import data
from typing import Tuple
from .train_utils import parse_args
from datasets import dataset_factory
import yaml
import torch
import pdb
from scipy.ndimage import minimum_filter, maximum_filter, uniform_filter
from time import time

def compute_local_min_max_avg(image: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    footprint = np.ones((window_size, window_size))
    
    local_min_map = minimum_filter(image, footprint=footprint, mode='reflect')
    local_max_map = maximum_filter(image, footprint=footprint, mode='reflect')
    local_avg_map = uniform_filter(image, size=window_size, mode='reflect')
    
    return local_min_map, local_max_map, local_avg_map

def apply_transfer_function(image: np.ndarray, local_min_map: np.ndarray, local_max_map: np.ndarray, local_avg_map: np.ndarray) -> np.ndarray:
    enhanced_image = np.zeros_like(image, dtype=np.float32)
    
    lmin = local_min_map.astype(np.float32)
    lmax = local_max_map.astype(np.float32)
    avg = local_avg_map.astype(np.float32)
    window = lmax - lmin
    window = np.sqrt(np.clip(window * (510 - window), 0, None))
    
    valid = lmin != lmax
    lmax_lmin_diff = np.where(valid, lmax - lmin, 1)  # Avoid division by zero
    img = np.where(valid, window * (image - lmin) / lmax_lmin_diff, image)
    avg = np.where(valid, window * (avg - lmin) / lmax_lmin_diff, avg)
    window_nonzero = np.where(window != 0, window, 1)  # Avoid division by zero
    alpha = np.where(window != 0, (avg - img) / (181.019 * window_nonzero), 0)
    
    a = 0.707 * alpha
    b = 1.414 * alpha * (img - window) - 1
    c = 0.707 * alpha * img * (img - 2 * window) + img
    
    discriminant = b * b - 4 * a * c
    sqrt_discriminant = np.sqrt(np.clip(discriminant, 0, None))
    valid_alpha = (alpha != 0) & (discriminant >= 0)
    a_nonzero = np.where(a != 0, a, 1) # Avoid division by zero
    enhanced_image = np.where(valid_alpha, lmin + (-b - sqrt_discriminant) / (2 * a_nonzero), img + lmin)
    return np.clip(enhanced_image, 0, 255).astype(np.uint8)

def anisotropic_propagation(image: np.ndarray, conductivity: float) -> np.ndarray:
    return gaussian_filter(image, sigma=conductivity)

def contrast_enhancement_band(image_band: np.ndarray, window_size: int, conductivity: float) -> np.ndarray:
    local_min_map, local_max_map, local_avg_map = compute_local_min_max_avg(image_band, window_size)
    local_min_map = anisotropic_propagation(local_min_map, conductivity)
    local_max_map = anisotropic_propagation(local_max_map, conductivity)
    local_avg_map = anisotropic_propagation(local_avg_map, conductivity)
    enhanced_band = apply_transfer_function(image_band, local_min_map, local_max_map, local_avg_map)
    return enhanced_band

def contrast_enhancement_multispectral(image: np.ndarray, window_size: int, conductivity: float) -> np.ndarray:
    enhanced_image = np.zeros_like(image)
     # Process each band independently
    for band in range(image.shape[-1]):
        enhanced_image[:, :, band] = contrast_enhancement_band(image[:, :, band], window_size, conductivity)
    return enhanced_image


if __name__ == '__main__':
    conductivity = 0.95
    window_size = 5
    plot = True
    # test_plot_rgb()
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    torch.cuda.set_device(config['device'])
    dataset_name = config['dataset']['name']
    train_dataset = dataset_factory[config['dataset']['name']](
                    **config['dataset']['kwargs'], mode="train", 
                    transforms=None)
    sri, orig_rgb, gt = train_dataset.img_sri, train_dataset.img_rgb, train_dataset.gt
    # show original RGB
    blurred_sri = train_dataset.downsample(sri)
    blurred_sri = (blurred_sri*255).astype(np.uint8)
    blurred_rgb = train_dataset.get_rgb(blurred_sri)
    # show blurred RGB extracted from blurred sri
    st = time()
    enhanced_sri = contrast_enhancement_multispectral(blurred_sri, 
                                        window_size, conductivity)
    end = time() - st
    print(f"Time taken: {end:.2f} seconds ({end * 1000:.2f} milliseconds)")
    enhanced_rgb = train_dataset.get_rgb(enhanced_sri)
    
    if plot:
        # Plot the original, blurred, and enhanced images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax = axes.ravel()
        ax[0].imshow(orig_rgb)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(blurred_rgb)
        ax[1].set_title('Blurred Image')
        ax[1].axis('off')
        ax[2].imshow(enhanced_rgb)
        ax[2].set_title('Enhanced Image')
        ax[2].axis('off')
        # Save the plot
        plt.savefig('images/contrast_enhancement_results_jasper_ridge_parallel.png')
        plt.show()
