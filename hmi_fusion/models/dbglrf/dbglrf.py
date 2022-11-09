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
from .tv_layers_for_cv.tv_opt_layers.layers.general_tv_2d_layer import GeneralTV2DLayer
import scipy
dtype = torch.FloatTensor
from torchmetrics import SpectralAngleMapper
def calc_sam_loss(pred, tgt):
    sam = SpectralAngleMapper()
    return sam(pred, tgt)# * (180 / torch.pi)
    # return torch.arccos((pred.view(pred.shape[0], -1) * tgt.view(tgt.shape[0], -1)).sum(-1)/
    # (torch.linalg.norm(pred)*torch.linalg.norm(tgt))).mean() * (1 / torch.pi)
# def calc_sam_loss(pred, tgt):
#     p = pred.view(pred.shape[0], -1)
#     t = tgt.view(tgt.shape[0], -1)
#     # num = (pred * tgt)# B, N
#     # denom = torch.linalg.norm(pred, axis=1)*torch.linalg.norm(tgt, axis=1) # B, 1
#     # pdb.set_trace()
#     return (torch.arccos((p * t)/
#     ((torch.linalg.norm(p, axis=1)*torch.linalg.norm(t, axis=1))[:,None])).mean(-1) * 
#     (180 / torch.pi)).mean()

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.e_conv_1 = nn.Conv2d(in_channels, in_channels*2, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(in_channels*2)
        self.e_conv_2 = nn.Conv2d(in_channels*2, in_channels*2, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_2 = nn.BatchNorm2d(in_channels*2)
        self.e_conv_3 = nn.Conv2d(in_channels*2, out_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_3 = nn.BatchNorm2d(out_channels)
        

    def forward(self, x):
        #h1
        x = self.e_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_1(x)
        #h2
        x = self.e_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)    
        x = self.e_bn_2(x)     
        #h3
        x = self.e_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
        x = self.e_bn_3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dec_channels):
        super(Decoder, self).__init__()
        self.d_conv_1 = nn.Conv2d(dec_channels, dec_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.d_bn_1 = nn.BatchNorm2d(dec_channels)
        self.d_conv_2 = nn.Conv2d(dec_channels*3, dec_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.d_bn_2 = nn.BatchNorm2d(dec_channels)
        self.d_conv_3 = nn.Conv2d(dec_channels*3, dec_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        # self.tv_layer = GeneralTV2DLayer(lmbd_init=30,num_iter=1) #lmbd_init=30,num_iter=10
        self.d_bn_3 = nn.BatchNorm2d(dec_channels)
        

    def forward(self, x, layers):
        # h2
        x = F.interpolate(x, scale_factor=2)
        x = self.d_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_1(x)
        x = torch.cat([x, layers[0]], 1)
        # pdb.set_trace()
        # h3
        x = F.interpolate(x, scale_factor=2)
        x = self.d_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_2(x)
        x = torch.cat([x, layers[1]], 1)
        # h4
        x = F.interpolate(x, scale_factor=2)
        x = self.d_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        # x = self.tv_layer(x)
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
        # self.tv_layer = GeneralTV2DLayer(lmbd_init=30,num_iter=10)

    def forward(self, x):  
        #h1
        x = self.e_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_1(x)
        x = self.pool(x)
        o1 = x
        #h2
        x = self.e_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)    
        x = self.e_bn_2(x)     
        x = self.pool(x)
        o2 = x
        #h3
        x = self.e_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
        # x = self.tv_layer(x)
        x = self.e_bn_3(x)
        x = self.pool(x)
        return x, [o2, o1]
       

        

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, dec_channels, R):
        super(AutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.R = nn.Parameter(R)
        self.r_encoder = CNN(in_channels, in_channels)
        # self.x_encoder = nn.Sequential(
        # nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # nn.LeakyReLU(0.2),
        # nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # nn.LeakyReLU(0.2)
        # )
        self.encoder = Encoder(in_channels)
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

    def forward(self, x_k, z):
        B, C, Isize, Isize = z.shape
        # pdb.set_trace()
        z_aug = (self.R).T @ z.reshape(B, C, -1)
        z_aug = z_aug.reshape(z_aug.shape[0], z_aug.shape[1], Isize, Isize)
        # o = self.x_encoder(x_k) + self.r_encoder(z_aug)
        x_k += self.r_encoder(z_aug)
        # x_k = self.r_encoder(torch.cat([x_k, z], 1))
        y_hat, layers = self.encoder(x_k) # downsample from 512 to 64
        x_hat = self.decoder(y_hat, layers) # 31 channels
        return y_hat, x_hat
        

    def calc_loss(self, x_new, y_hat, lz, x_old, y):
        loss = nn.MSELoss()
        recon_loss_y = loss(y_hat, y)
        recon_loss_x = loss(x_new, x_old)
        # SAM loss
        sam_loss = calc_sam_loss(x_new, x_old) + calc_sam_loss(y_hat, y)
        # calc_sam_loss(y_new, x_old)
        ### Graph Laplacian Loss
        x_new = x_new.reshape(x_new.shape[0], x_new.shape[1], -1)
        lz_xt = linear_operator.utils.sparse.bdsmm(lz, x_new.transpose(1,2))
        GL = MatmulLinearOperator(x_new, lz_xt)
        factor = x_new.shape[-1]
        GL = torch.diagonal(to_dense(GL)/factor, dim1=-2, dim2=-1).sum() # trace
        
        return recon_loss_x, sam_loss, recon_loss_y, GL




class AutoEncoder2(nn.Module):
    def __init__(self, y_channel, z_channel):
        super(AutoEncoder2, self).__init__()
        self.y_channel = y_channel
        self.z_channel = z_channel
        self.decoder = Decoder(y_channel)
        self.cnn = CNN(y_channel+z_channel, y_channel)
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

    def forward(self, y, z):
        x_h = self.decoder(y)
        x_h_opt = self.cnn(torch.cat([x_h, z], 1))
        return x_h, x_h_opt

    def calc_loss(self, x1, x2, lz, x):
        loss = nn.MSELoss()
        recon_loss_x1 = loss(x1, x)
        recon_loss_x2 = loss(x2, x)
        # SAM loss
        sam_loss = (calc_sam_loss(x2, x) + calc_sam_loss(x1, x))/2
        ### Graph Laplacian Loss
        x2 = x2.reshape(x2.shape[0], x2.shape[1], -1)
        lz_xt = linear_operator.utils.sparse.bdsmm(lz, x2.transpose(1,2))
        GL = MatmulLinearOperator(x2, lz_xt)
        factor = x2.shape[-1]
        GL = torch.diagonal(to_dense(GL)/factor, dim1=-2, dim2=-1).sum() # trace
        return recon_loss_x1, recon_loss_x2, sam_loss, GL
        


class AutoEncoder3(nn.Module):
    def __init__(self, in_channels, dec_channels):
        super(AutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.encoder = Encoder(in_channels)
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

    def forward(self, x_k):
        y_hat, o1, o2 = self.encoder(x_k) # downsample from 512 to 64
        x_hat = self.decoder(y_hat) # 31 channels
        return y_hat, x_hat
        

    def calc_loss(self, x_new, y_hat, lz, x_old, y):
        loss = nn.MSELoss()
        recon_loss_y = loss(y_hat, y)
        recon_loss_x = loss(x_new, x_old)
        # SAM loss
        sam_loss = calc_sam_loss(x_new, x_old)
        ### Graph Laplacian Loss
        x_new = x_new.reshape(x_new.shape[0], x_new.shape[1], -1)
        lz_xt = linear_operator.utils.sparse.bdsmm(lz, x_new.transpose(1,2))
        GL = MatmulLinearOperator(x_new, lz_xt)
        factor = x_new.shape[-1]
        GL = torch.diagonal(to_dense(GL)/factor, dim1=-2, dim2=-1).sum() # trace
        return recon_loss_x, sam_loss, recon_loss_y, GL


class AutoEncoder3(nn.Module):
    def __init__(self, in_channels, dec_channels):
        super(AutoEncoder3, self).__init__()
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.downsampler = Encoder(in_channels)
        self.upsampler = Decoder(dec_channels)
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

    def forward(self, y):
        x_hat  = self.upsampler(y)
        # y_hat, o1, o2 = self.encoder(x_k) # downsample from 512 to 64
        y_hat = self.downsampler(x_hat) # 31 channels
        return y_hat, x_hat
        

    def calc_loss(self, x_hat, y_hat, lz, y):
        loss = nn.MSELoss()
        recon_loss_y = loss(y_hat, y)
        # SAM loss
        sam_loss = calc_sam_loss(y_hat, y)
        ### Graph Laplacian Loss
        x_hat = x_hat.reshape(x_hat.shape[0], x_hat.shape[1], -1)
        lz_xt = linear_operator.utils.sparse.bdsmm(lz, x_hat.transpose(1,2))
        GL = MatmulLinearOperator(x_hat, lz_xt)
        factor = x_hat.shape[-1]
        GL = torch.diagonal(to_dense(GL)/factor, dim1=-2, dim2=-1).sum() # trace
        return sam_loss, recon_loss_y, GL


# x = torch.rand(31, 512, 512)
# y = torch.rand(31, 64, 64)
# z = torch.rand(512, 512, 3).numpy()
# model = AutoEncoder(31, 31)
# y_hat, x_hat = model(x[None, ...])
# lz = getLaplacian(z, z.shape[2])

# xhd = x_hat.reshape(x_hat.shape[0], x_hat.shape[1], -1)
# GL = torch.sum((torch.diagonal(xhd)**2) * torch.diagonal(lz))