import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
    
    
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def calculate_psnr(block1, block2):
    mse = np.mean((block1 - block2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
    
    
class FocalDice(nn.Module):
    def __init__(self, alpha=1.0, gamma=3.0):
        super(FocalDice, self).__init__()
        self.alpha = alpha
        self.loss = FocalLoss(gamma=gamma)

    def forward(self, inputs, targets):
        loss = self.loss(inputs['preds'], targets)
        dice = dice_loss(inputs['preds'], targets)
        return self.alpha * loss + (1 - self.alpha) * dice
 
 
class Focal(nn.Module):
    def __init__(self, gamma=3.0):
        super().__init__()
        self.loss = FocalLoss(gamma=gamma)

    def forward(self, inputs, targets):
        loss = self.loss(inputs['preds'], targets)
        return loss 
     
 
 
def spectral_angle_mapper_loss(embedding1, embedding2):
    """
    Calculate the spectral angle mapper (SAM) loss between two embeddings.
    Args:
        embedding1: Tensor of shape (B, C, H, W) - First embedding
        embedding2: Tensor of shape (B, C, H, W) - Second embedding
    Returns:
        loss: SAM loss
    """
    dot_product = torch.sum(embedding1 * embedding2, dim=1)
    norm1 = torch.norm(embedding1, p=2, dim=1)
    norm2 = torch.norm(embedding2, p=2, dim=1)
    cos_theta = dot_product / (norm1 * norm2 + 1e-8)
    loss = torch.mean(torch.acos(torch.clamp(cos_theta, -1 + 1e-8, 1 - 1e-8)))
    return loss   

class FocalSAM(nn.Module):
    def __init__(self, alpha=1.0, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.loss = FocalLoss(gamma=gamma)

    def forward(self, inputs, targets):
        ipts = inputs['preds']
        z_hsi, z_msi = inputs['embeddings']
        sam = spectral_angle_mapper_loss(z_hsi, z_msi)
        loss = self.loss(ipts, targets)
        return self.alpha * loss + (1 - self.alpha) * sam
    


loss_factory = {
    'focal_sam': FocalSAM,
    'focal': Focal,
    'focal_dice': FocalDice
}