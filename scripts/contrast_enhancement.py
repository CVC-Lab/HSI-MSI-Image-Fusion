import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Callable
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import data
from .train_utils import parse_args
from datasets import dataset_factory
import yaml
import torch
import pdb


def compute_local_min_max_avg(image: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    
    local_min_map = np.zeros_like(image)
    local_max_map = np.zeros_like(image)
    local_avg_map = np.zeros_like(image)
    
    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            local_window = padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            local_min_map[i-pad_size, j-pad_size] = np.min(local_window)
            local_max_map[i-pad_size, j-pad_size] = np.max(local_window)
            local_avg_map[i-pad_size, j-pad_size] = np.mean(local_window)
    
    return local_min_map, local_max_map, local_avg_map


def apply_transfer_function(image: np.ndarray, local_min_map: np.ndarray, local_max_map: np.ndarray, local_avg_map: np.ndarray) -> np.ndarray:
    enhanced_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lmin = local_min_map[i, j]
            lmax = local_max_map[i, j]
            avg = local_avg_map[i, j]
            window = lmax - lmin
            window = np.sqrt(window * (510 - window))
            
            if lmin != lmax:
                img = window * (image[i, j] - lmin) / (lmax - lmin)
                avg = window * (avg - lmin) / (lmax - lmin)
            else:
                img = image[i, j]

            alpha = (avg - img) / (181.019 * window) if window != 0 else 0
            if alpha != 0:
                a = 0.707 * alpha
                b = 1.414 * alpha * (img - window) - 1
                c = 0.707 * alpha * img * (img - 2 * window) + img
                enhanced_image[i, j] = lmin + (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
            else:
                enhanced_image[i, j] = img + lmin

            enhanced_image[i, j] = np.clip(enhanced_image[i, j], 0, 255)  # Ensure the values are within the valid range
    return enhanced_image

def anisotropic_propagation(image: np.ndarray, conductivity: float) -> np.ndarray:
    filtered_image = gaussian_filter(image, sigma=conductivity)
    return filtered_image

def contrast_enhancement(image: np.ndarray, window_size: int, conductivity: float) -> np.ndarray:
    local_min_map, local_max_map, local_avg_map = compute_local_min_max_avg(image, window_size)
    local_min_map = anisotropic_propagation(local_min_map, conductivity)
    local_max_map = anisotropic_propagation(local_max_map, conductivity)
    local_avg_map = anisotropic_propagation(local_avg_map, conductivity)
    enhanced_image = apply_transfer_function(image, 
                                             local_min_map, 
                                             local_max_map, 
                                             local_avg_map)
    return enhanced_image

def contrast_enhancement_band(image_band: np.ndarray, window_size: int, conductivity: float) -> np.ndarray:
    local_min_map, local_max_map, local_avg_map = compute_local_min_max_avg(image_band, window_size)
    local_min_map = anisotropic_propagation(local_min_map, conductivity)
    local_max_map = anisotropic_propagation(local_max_map, conductivity)
    local_avg_map = anisotropic_propagation(local_avg_map, conductivity)
    enhanced_band = apply_transfer_function(image_band, 
                                            local_min_map, 
                                            local_max_map, 
                                            local_avg_map)
    return enhanced_band

def contrast_enhancement_multispectral(image: np.ndarray, window_size: int, conductivity: float) -> np.ndarray:
    # Initialize an empty array for the enhanced image with the same shape as the input
    enhanced_image = np.zeros_like(image)
    
    # Process each band independently
    for band in range(image.shape[-1]):
        enhanced_image[:, :, band] = contrast_enhancement_band(image[:, :, band], window_size, conductivity)
    
    return enhanced_image


def test_plot_rgb(conductivity = 0.95, window_size = 5):
    # Example usage
    image = data.camera()  # Load an example image
    image = image.astype(np.uint8)
    # Blur the image
    blurred_image = gaussian_filter(image, sigma=2)
    # Enhance the image
    enhanced_image = contrast_enhancement(image, window_size, conductivity)
    # Plot the original, blurred, and enhanced images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes.ravel()
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(blurred_image, cmap='gray')
    ax[1].set_title('Blurred Image')
    ax[1].axis('off')
    ax[2].imshow(enhanced_image, cmap='gray')
    ax[2].set_title('Enhanced Image')
    ax[2].axis('off')
    # Save the plot
    plt.savefig('contrast_enhancement_results.png')
    plt.show()

if __name__ == '__main__':
    conductivity = 0.95
    window_size = 5
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
    enhanced_sri = contrast_enhancement_multispectral(blurred_sri, 
                                        window_size, conductivity)
    enhanced_rgb = train_dataset.get_rgb(enhanced_sri)
    
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
    plt.savefig('images/contrast_enhancement_results_jasper_ridge.png')
    plt.show()
    