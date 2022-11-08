import os
from time import time
# import numpy as np
# import pandas as pd
# from PIL import Image
# import matplotlib.pyplot as plt
import linear_operator
from linear_operator.operators import MatmulLinearOperator
from linear_operator import to_dense
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy.sparse as ss
import numpy as np
import pdb
from ..hip.fusion.getLaplacian import getLaplacian
import scipy
dtype = torch.FloatTensor

class Decoder(nn.Module):
    def __init__(self, dec_channels):
        super(Decoder, self).__init__()
        self.d_conv_1 = nn.Conv2d(dec_channels, dec_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.d_bn_1 = nn.BatchNorm2d(dec_channels)
        self.d_conv_2 = nn.Conv2d(dec_channels, dec_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.d_bn_2 = nn.BatchNorm2d(dec_channels)
        self.d_conv_3 = nn.Conv2d(dec_channels, dec_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.d_bn_3 = nn.BatchNorm2d(dec_channels)

    def forward(self, x):
        # h2
        x = F.interpolate(x, scale_factor=2)
        x = self.d_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_1(x)
        # h3
        x = F.interpolate(x, scale_factor=2)
        x = self.d_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_2(x)
        # h4
        x = F.interpolate(x, scale_factor=2)
        x = self.d_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_3(x)  
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=512) -> None:
        super(Encoder, self).__init__()
        self.e_conv_1 = nn.Conv2d(in_channels, in_channels*2, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(in_channels*2)
        self.e_conv_2 = nn.Conv2d(in_channels*2, in_channels*2, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_2 = nn.BatchNorm2d(in_channels*2)
        self.e_conv_3 = nn.Conv2d(in_channels*2, in_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_3 = nn.BatchNorm2d(in_channels)
        self.pool = nn.MaxPool2d(2)


    def forward(self, x):   
        #h1
        x = self.e_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_1(x)
        x = self.pool(x)
        
        #h2
        x = self.e_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)    
        x = self.e_bn_2(x)     
        x = self.pool(x)
        #h3
        x = self.e_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
        x = self.e_bn_3(x)
        x = self.pool(x)

        return x
       

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, dec_channels):
        super(AutoEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        ###############
        # ENCODER
        ##############
        self.encoder = Encoder(in_channels)
        ###############
        # DECODER
        ##############
        self.decoder = Decoder(dec_channels)
        # Reinitialize weights using He initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def forward(self, x):
        y_hat = self.encoder(x) # downsample from 512 to 64
        x_hat = self.decoder(y_hat) # 31 channels
        return y_hat, x_hat

def calc_sam_loss(pred, tgt):
    return torch.arccos((pred * tgt)/(torch.linalg.norm(pred)*torch.linalg.norm(tgt))).mean()

def calc_loss(x_hat, y_hat, lz, x, y):
    loss = nn.MSELoss()
    recon_loss_y = loss(y_hat, y)
    recon_loss_x = loss(x_hat, x)
    # SAM loss
    # sam_loss = sam(x_hat, x)
    sam_loss = calc_sam_loss(x_hat, x)
    ### Graph Laplacian Loss
    # pdb.set_trace()
    # x_hat = x_hat * transform.std[None, :, None, None] + transform.mean[None, :, None, None]
    x_hat = x_hat.reshape(x_hat.shape[0], x_hat.shape[1], -1)
    lz_xt = linear_operator.utils.sparse.bdsmm(lz, x_hat.transpose(1,2))
    GL = MatmulLinearOperator(x_hat, lz_xt)
    factor = x_hat.shape[-1]
    GL = torch.diagonal(to_dense(GL)/factor, dim1=-2, dim2=-1).sum() # trace
    return recon_loss_x, sam_loss, recon_loss_y, GL




# x = torch.rand(31, 512, 512)
# y = torch.rand(31, 64, 64)
# z = torch.rand(512, 512, 3).numpy()
# model = AutoEncoder(31, 31)
# y_hat, x_hat = model(x[None, ...])
# lz = getLaplacian(z, z.shape[2])

# xhd = x_hat.reshape(x_hat.shape[0], x_hat.shape[1], -1)
# GL = torch.sum((torch.diagonal(xhd)**2) * torch.diagonal(lz))