import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from losses import CombinedLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from neural_nets.ca_siamese_unet import CASiameseUNet
from datasets import SingleImageDataset
from torch.utils.data import DataLoader
from einops import rearrange, pack
import pdb


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        B, C, H, W = inputs.shape
        inputs = rearrange(inputs, 'B C H W -> B C (H W)')
        targets = rearrange(targets, 'B C H W -> B C (H W)')
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        F_loss = F_loss.reshape(B, C, -1)
        return F_loss.mean(2)
    
loss_fn = FocalLoss()
# loss_fn = nn.MSELoss(reduction='none')

def visualize_jacobian_bar_plot(trainloader, model,
                       save_path='models/trained_model.pth', device=DEVICE):
    jacobian_sum = 0
    for i, data in tqdm(enumerate(trainloader, 0)):
        hsi_batch, rgb_batch, labels_batch = data
        hsi_batch.requires_grad = True
        rgb_batch.requires_grad = False
        jacobian = torch.autograd.functional.jacobian(
                                        lambda x: loss_fn(model(x, rgb_batch.to(device)), labels_batch.to(device)), #outputs.mean((2, 3))
                                        (hsi_batch.to(device)), 
                                        strict=True)
        jacobian_sum += jacobian.squeeze().mean((0, 2, 4, 5))
    avg_jacobian = jacobian_sum / len(trainloader) # [4, 6]
    avg_jacobian = avg_jacobian.cpu().numpy()
    # x label -> class
    # y label -> bar plot of each of the 6 channels showing their importance
    x_labels = ["road", "soil", "water", "tree"]
    y_labels = [f"c_{i}" for i in range(1, 7)]
    # Plotting the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.15  # the width of the bars
    x = np.arange(len(x_labels))
    
    for i, y_label in enumerate(y_labels):
        ax.bar(x + i*width, avg_jacobian[:, i], width, label=y_label)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Average Importance')
    ax.set_title('Average Importance of Channels for Each Class')
    ax.set_xticks(x + width * (len(y_labels) - 1) / 2)
    ax.set_xticklabels(x_labels)
    ax.legend()
    
    # Save the plot
    plt.savefig('jacobian_jasper.png')
    plt.show()
    
if __name__ == '__main__':
    img_path = 'data/jasper/jasper_ridge_224.mat'
    gt_path = 'data/jasper/jasper_ridge_gt.mat'
    start_band = 380
    end_band = 2500
    rgb_width = 64 
    rgb_height = 64
    hsi_width = 32 
    hsi_height = 32
    channels=[20, 60, 80, 100, 120, 140]
    save_path = 'models/trained_ca_siamese_model_final.pth'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CASiameseUNet(6, 3, 256, 4)
    model.load_state_dict(torch.load(save_path))
    model.to(torch.double).to(DEVICE)
    train_dataset = SingleImageDataset(channels,
                    img_path, gt_path,
                    start_band, end_band, 
                    rgb_width, rgb_height,
                    hsi_width, hsi_height, mode="train", 
                    transforms=None)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    visualize_jacobian_bar_plot(model=model, trainloader=train_loader)