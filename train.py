import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from neural_nets.models import SiameseUNet
from datasets import SingleImageDataset
from train_utils import main_training_loop, test

img_path = 'data/jasper/jasper_ridge_224.mat'
gt_path = 'data/jasper/jasper_ridge_gt.mat'
start_band = 380
end_band = 2500
rgb_width = 64 
rgb_height = 64
hsi_width = 32 
hsi_height = 32
channels=[20, 60, 80, 100, 120, 140]

train_dataset = SingleImageDataset(channels,
                 img_path, gt_path,
                 start_band, end_band, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height, mode="train")
test_dataset = SingleImageDataset(channels,
                 img_path, gt_path,
                 start_band, end_band, 
                 rgb_width, rgb_height,
                 hsi_width, hsi_height, mode="test")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SiameseUNet(6, 3, 256, 4).to(torch.double).to(DEVICE)
optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode='min', factor=0.5, patience=3)
# main_training_loop(train_loader, net, optimizer, scheduler,
#                  num_epochs=40, device=DEVICE, log_interval=40)

mIOU, gdice = test(test_loader, net, save_path='models/trained_model.pth', num_classes=4)
print(f"mIOU: {mIOU}, gdice: {gdice}")
 