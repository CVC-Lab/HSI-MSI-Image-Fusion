import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from einops import rearrange
import pdb

# Define albumentations transforms
albumentations_transform = A.Compose([
    # flips
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # color jitter
    # A.RandomBrightnessContrast(p=0.5),
    # extra noise
    A.GaussNoise(p=0.5),
    # A.ElasticTransform(p=0.5),
    # non-rigid transforms
    A.OneOf([
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(p=0.5),
    ], p=0.5),
    # non spatial transforms
    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8),
    ToTensorV2()
], additional_targets={'msi': 'image', 'gt': 'mask'},
is_check_shapes=False)



class AddSingleScattering(iaa.meta.Augmenter):
    def __init__(self, beta=0.1, A=0.8, depth_method='linear', name=None, deterministic=False, random_state=None):
        super(AddSingleScattering, self).__init__(name=name, random_state=random_state)
        self.beta = beta
        self.A = A
        self.depth_method = depth_method

    def generate_synthetic_depth_map(self, image_shape):
        if self.depth_method == 'linear':
            depth_map = np.tile(np.linspace(0, 1, image_shape[1]), (image_shape[0], 1))
        elif self.depth_method == 'random':
            depth_map = np.random.uniform(0, 1, image_shape)
        else:
            raise ValueError("Unknown method for depth map generation.")
        return depth_map

    def add_single_scattering(self, image, depth_map):
        transmission_map = np.exp(-self.beta * depth_map)
        hazy_image = image * transmission_map[:, :, np.newaxis] + self.A * (1 - transmission_map[:, :, np.newaxis])
        return hazy_image
        # return (hazy_image * 255).astype(np.uint8)

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        for image in images:
            depth_map = self.generate_synthetic_depth_map(image.shape[:2])
            hazy_image = self.add_single_scattering(image, depth_map)
            result.append(hazy_image)
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def get_parameters(self):
        return [self.beta, self.A, self.depth_method]

# Example usage in an imgaug pipeline
augmentation_pipeline = iaa.Sequential([
    iaa.MotionBlur(k=15, angle=[-45, 45]),
    # iaa.GammaContrast(1.5, per_channel=True),
    AddSingleScattering(beta=0.1, A=0.8, depth_method='random')
])

# Example application
def apply_augmentation(hsi, gt, get_rgb, downsample, conductivity=0.95, window_size=3):
    hsi_aug = augmentation_pipeline(image=hsi)
    msi_aug = get_rgb(hsi_aug)
    hsi_aug = downsample(hsi_aug)
    hsi_aug = rearrange(hsi_aug, "H W C -> C H W")
    msi_aug = rearrange(msi_aug, "H W C -> C H W")
    gt = rearrange(gt, "H W C -> C H W")
    return hsi_aug, msi_aug, gt



# Apply transformations to an image
def apply_transforms(hsi, msi, gt):
    transformed = albumentations_transform(image=hsi, msi=msi, gt=gt)
    transformed['gt'] = rearrange(transformed['gt'], "H W C -> C H W")
    return transformed['image'], transformed['msi'], transformed['gt']


if __name__ == '__main__':
    hsi = torch.rand(64, 64, 31).numpy()
    msi = torch.rand(256, 256, 3).numpy()
    gt = torch.rand(256, 256, 5).numpy()
    hsi, msi, gt = apply_augmentation(hsi, msi, gt)
    # hsi, msi, gt = apply_transforms(hsi, msi, gt)
    print(hsi.shape, msi.shape, gt.shape)