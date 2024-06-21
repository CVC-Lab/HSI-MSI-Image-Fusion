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

def calculate_psnr(block1, block2, num_classes):
    mse = np.mean((block1 - block2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = num_classes - 1 
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
    
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=3.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        # self.cross_entropy = nn.CrossEntropyLoss()
        self.loss = FocalLoss(gamma=gamma)

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        dice = dice_loss(inputs, targets)
        return self.alpha * loss + (1 - self.alpha) * dice