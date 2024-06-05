import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2


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

# Apply transformations to an image
def apply_transforms(hsi, msi, gt):
    transformed = albumentations_transform(image=hsi, msi=msi, gt=gt)
    return transformed['image'], transformed['msi'], transformed['gt']


if __name__ == '__main__':
    hsi = torch.rand(64, 64, 31).numpy()
    msi = torch.rand(256, 256, 3).numpy()
    gt = torch.rand(256, 256, 5).numpy()
    hsi, msi, gt = apply_transforms(hsi, msi, gt)
    print(hsi.shape, msi.shape, gt.shape)