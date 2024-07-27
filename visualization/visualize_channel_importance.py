import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from losses import CombinedLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops import rearrange, pack
from neural_nets import model_factory
from datasets import dataset_factory
from transforms import apply_augmentation
import seaborn as sns
import argparse
import yaml
import pdb
import random

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

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

def visualize_all_channel_importance_bar_plot(model, 
                                              trainloader,
                                              config=None, 
                                             device=None):
    channels_imp_avg = torch.zeros(train_loader.dataset.img_sri.shape[-1])
    channels_imp_avg = channels_imp_avg.to(device)
    for i, data in tqdm(enumerate(trainloader, 0)):
        hsi_batch, rgb_batch, labels_batch = data
        _ = model.encoder.channel_selector(hsi_batch.to(device))
        channel_wts = model.encoder.channel_selector.importance_wts
        channels_imp_avg += channel_wts.squeeze().mean(0)
        
    channels_imp_avg /= len(trainloader)
    channels_imp_avg = F.softmax(channels_imp_avg, 0).detach().cpu().numpy()
    
    # Print top 6 channel indexes
    top_channels = np.argsort(channels_imp_avg)[-6:][::-1]
    print("Top 6 channel indexes based on importance weights:", sorted(top_channels))
    # # Plotting the channel importance
    plt.figure(figsize=(12, 6))
    sns.heatmap(channels_imp_avg.reshape(1, -1), cmap='viridis')
    
    # plt.axis([0, len(channels_imp_avg), channels_imp_avg.min(), channels_imp_avg.max()]) 
    # plt.bar(range(len(channels_imp_avg)), channels_imp_avg, color='skyblue')
    plt.xlabel('Channel Index')
    plt.ylabel('Average Importance Weight')
    plt.title('Channel Importance Bar Plot')
    plt.savefig(f"{config['model']['name']}_{config['dataset']['name']}_IWCA.png")
    plt.show()
    

def visualize_jacobian_bar_plot(model, trainloader,
                       config, select_channels=None, device=None):
    loss_fn = FocalLoss()
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
    # y label -> bar plot of each of the channels showing their importance
    x_labels = config['dataset']['kwargs']['classes']
    if not select_channels:
        select_channels = range(config['model']['kwargs']['hsi_in'])    
    y_labels = [f"ch_{ch}" for ch in select_channels]
    # Plotting the bar plot
    fig, ax = plt.subplots(figsize=(len(y_labels), 12))
    width = 0.15  # the width of the bars
    x = np.arange(len(x_labels))
    
    for i, y_label in enumerate(y_labels):
        ax.bar(x + i*width, 
               avg_jacobian[:, select_channels[i]], width, label=y_label)
        
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Average Importance')
    ax.set_title('Average Importance of Channels for Each Class')
    ax.set_xticks(x + width * (len(y_labels) - 1) / 2)
    ax.set_xticklabels(x_labels)
    ax.legend()
    
    # Save the plot
    plt.savefig(f"{config['model']['name']}_{config['dataset']['name']}_class_channel_importance.png")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Run deep learning experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    torch.cuda.set_device(config['device'])
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    save_path = f'models/trained_{model_name}_{dataset_name}_final_noisy.pth'
    train_dataset = dataset_factory[config['dataset']['name']](
                    **config['dataset']['kwargs'], mode="train", 
                    transforms=apply_augmentation)
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['dataset']['batch_size'], 
                              shuffle=True)
    DEVICE = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    model = model_factory[model_name](**config['model']['kwargs']).to(torch.double).to(DEVICE)
    # visualize_jacobian_bar_plot(model=model, 
    #                             trainloader=train_loader, 
    #                             config=config, device=DEVICE)
    model.eval()
    visualize_all_channel_importance_bar_plot(model=model, 
                                trainloader=train_loader, 
                                config=config, device=DEVICE)