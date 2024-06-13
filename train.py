import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from neural_nets.siamese_unet import SiameseUNet
from neural_nets.sam_siamese_unet import SamSiameseUNet
from neural_nets.ca_siamese_unet import CASiameseUNet
from neural_nets.basic_model import CNN
from neural_nets.unet import UNet
from datasets import SingleImageDataset
from train_utils import main_training_loop, test
from transforms import apply_transforms, apply_augmentation

torch.cuda.set_device(1)
# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# # Ensure deterministic behavior in PyTorch
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# # Set environment variables
# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for CUDA 10.2 and later
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# # Set number of threads
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
# Seed worker function
# def seed_worker(worker_id):
#     np.random.seed(42 + worker_id)
#     random.seed(42 + worker_id)

# # Generator for data loader
# g = torch.Generator()
# g.manual_seed(42)


img_path = 'data/jasper/jasper_ridge_224.mat'
gt_path = 'data/jasper/jasper_ridge_gt.mat'
start_band = 380
end_band = 2500
rgb_width = 64 
rgb_height = 64
hsi_width = 32 
hsi_height = 32
channels=[20, 60, 80, 100, 120, 140]
model_name = 'ca_siamese'

model_factory = {
    'ca_siamese': CASiameseUNet,
    'unet': UNet,
    'sam_siamese': SamSiameseUNet,
    'cnn': CNN
    
}
model_args = {
    'ca_siamese': (6, 3, 256, 4),
    'unet': (3, 256, 4),
    'sam_siamese': (6, 3, 256, 4),
    'cnn': (3, 4)
}
save_path = f'models/trained_{model_name}_final_noisy.pth'

train_dataset = SingleImageDataset(channels,
                 img_path, gt_path,
                 start_band, end_band, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height, mode="train", 
                 transforms=apply_augmentation)
test_dataset = SingleImageDataset(channels,
                 img_path, gt_path,
                 start_band, end_band, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height, mode="test", transforms=apply_augmentation)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model_factory[model_name](*model_args[model_name]).to(torch.double).to(DEVICE)
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode='min', factor=0.5, patience=3)
main_training_loop(train_loader, net, optimizer, scheduler, save_path=save_path,
                 num_epochs=40, device=DEVICE, log_interval=2)

mIOU, gdice = test(test_loader, net, save_path=save_path, num_classes=4)
print(f"mIOU: {mIOU}, gdice: {gdice}")
 