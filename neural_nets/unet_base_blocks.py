import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

# each down block reduces spatial dimensions // 2 and increases channel count by 2
class Conv1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=1)
        self.b1 = nn.BatchNorm2d(in_channels*2)
        self.c2 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, stride=2, padding=1)
        self.b2 = nn.BatchNorm2d(in_channels*2)
        self.c3 = nn.Conv2d(in_channels*2, out_channels, kernel_size=1)
        self.b3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.b1(F.relu(self.c1(x)))
        x = self.b2(F.relu(self.c2(x)))
        x = self.b3(F.relu(self.c3(x)))
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_pooled = self.pool(x)
        return x_pooled, x

class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.global_avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        avg_out = self.sigmoid(avg_out)
        return x * avg_out

class Down1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_block1 = DownBlock(in_channels, 64)
        self.down_block2 = DownBlock(64, 128)
        self.down_block3 = DownBlock(128, out_channels)
        self.spectral_attention = SpectralAttention(out_channels)
        
    def forward(self, x):
        x1_pooled, x1 = self.down_block1(x)
        x2_pooled, x2 = self.down_block2(x1_pooled)
        x3_pooled, x3 = self.down_block3(x2_pooled)
        x3_attention = self.spectral_attention(x3_pooled)
        return x3_attention, [x1, x2, x3]
