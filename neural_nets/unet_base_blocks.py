import torch.nn as nn
import torch.nn.functional as F
import torch


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