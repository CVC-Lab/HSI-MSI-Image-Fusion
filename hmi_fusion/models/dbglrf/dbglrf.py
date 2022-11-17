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
import torch.nn.utils.parametrize as parametrize
# from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import spectral_norm
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
import gpytorch
import math

def calc_sam_loss(pred, tgt):
    sam = SpectralAngleMapper()
    return torch.nan_to_num(sam(pred, tgt), nan=0.0, posinf=1.0)# * (180 / torch.pi)
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


class PositiveSparseConstraint(nn.Module):
    def forward(self, x):
        # eps = 1e-12
        nc, c, k, k = x.shape
        w = x.reshape(nc, c, -1)
        # return F.normalize(torch.abs(w), eps=eps, dim=-1).reshape(nc, c, k, k)
        # return F.normalize(torch.abs(w), eps=eps, dim=-1).reshape(nc, c, k, k)
        return F.softmax(w, -1).reshape(nc, c, k, k)


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
        # pdb.set_trace()
        # self.e_conv_1 = spectral_norm(self.e_conv_1)
        # self.e_conv_2 = spectral_norm(self.e_conv_2)
        # self.e_conv_3 = spectral_norm(self.e_conv_3)
        # parametrize.register_parametrization(self.e_conv_1, "weight", PositiveSparseConstraint())
        # parametrize.register_parametrization(self.e_conv_2, "weight", PositiveSparseConstraint())    
        # parametrize.register_parametrization(self.e_conv_3, "weight", PositiveSparseConstraint())           

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
        # parametrize.register_parametrization(self.d_conv_1, "weight", PositiveSparseConstraint())
        # parametrize.register_parametrization(self.d_conv_2, "weight", PositiveSparseConstraint())    
        # parametrize.register_parametrization(self.d_conv_3, "weight", PositiveSparseConstraint())     
        # self.d_conv_1 = spectral_norm(self.d_conv_1)
        # self.d_conv_2 = spectral_norm(self.d_conv_2)
        # self.d_conv_3 = spectral_norm(self.d_conv_3)
        

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
        # parametrize.register_parametrization(self.e_conv_1, "weight", PositiveSparseConstraint())
        # parametrize.register_parametrization(self.e_conv_2, "weight", PositiveSparseConstraint())    
        # parametrize.register_parametrization(self.e_conv_3, "weight", PositiveSparseConstraint())     
        # self.e_conv_1 = spectral_norm(self.e_conv_1)
        # self.e_conv_2 = spectral_norm(self.e_conv_2)
        # self.e_conv_3 = spectral_norm(self.e_conv_3)    
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
       

class SegCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.e_conv_1 = nn.Conv2d(in_channels, in_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(in_channels)
        self.e_conv_2 = nn.Conv2d(in_channels, in_channels, 
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_2 = nn.BatchNorm2d(in_channels)
        # self.e_conv_3 = nn.Conv2d(in_channels*2, out_channels, 
        #                           kernel_size=(3, 3), stride=(1, 1), padding=1)
        # self.e_bn_3 = nn.BatchNorm2d(out_channels)      

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
        # x = self.e_conv_3(x)
        # x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
        # x = self.e_bn_3(x)
        B, C, I, I = x.shape
        x = F.softmax(x.reshape(B, C, -1), dim=-1).reshape(B, C, I, I)
        return x
        


class Baseline(nn.Module):
    def __init__(self, in_channels, dec_channels, R):
        super(Baseline, self).__init__()
        self.R = R
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        # self.seg_cnn = SegCNN(in_channels, in_channels)
        self.e_conv_1 = nn.Conv2d(in_channels, in_channels, 
                    kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(in_channels)
        # self.cnn = CNN(in_channels, in_channels)
        # self.tv_layer = GeneralTV2DLayer(lmbd_init=30,num_iter=1)

            
    def forward(self, x_k, z, lz):
        B, C, Isize, Isize = x_k.shape
        _, zc, _, _ = z.shape
        # Zt
        Zt = (self.R).T @ z.reshape(B, zc, -1)
        Zt = Zt.reshape(Zt.shape[0], Zt.shape[1], Isize, Isize)
        # some bands have hri information and other are zero
        # LY = linear_operator.utils.sparse.bdsmm(lz, x_k.reshape(B, C, -1).transpose(1, 2))
        # LY = LY.reshape(B, C, Isize, Isize)
        soft_mask = self.seg_cnn(Zt)
        # s = torch.sum(Zt, dim=(2, 3))
        # soft_mask = s/s.max()
        # soft_mask = soft_mask[:, :, None, None].repeat(1, 1, Isize, Isize)
        x_o = Zt*soft_mask + x_k*(1-soft_mask)
        # x_o = self.cnn(x_o)

        x_o = self.e_conv_1(x_o)
        x_o = self.e_bn_1(x_o)
        x_o = F.leaky_relu(x_o, negative_slope=0.2, inplace=True)  

        # x_o = self.tv_layer(x_o)
        return x_o

    def calc_loss(self, x_new, lz, x_old):
        loss = nn.MSELoss()
        
        recon_loss_x = loss(x_new, x_old)
        # SAM loss
        sam_loss = calc_sam_loss(x_new, x_old) #+ calc_sam_loss(y_hat, y)
        # sam_loss = torch.tensor([0.0]).to(recon_loss_y.device)
        # calc_sam_loss(y_new, x_old)
        ### Graph Laplacian Loss
        x_new = x_new.reshape(x_new.shape[0], x_new.shape[1], -1)
        lz_xt = linear_operator.utils.sparse.bdsmm(lz, x_new.transpose(1,2))
        GL = MatmulLinearOperator(x_new, lz_xt)
        factor = x_new.shape[-1]
        GL = torch.diagonal(to_dense(GL)/factor, dim1=-2, dim2=-1).sum() # trace
        return recon_loss_x, sam_loss, GL




class AutoEncoder(nn.Module):
    def __init__(self, in_channels, dec_channels, R):
        super(AutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        # self.R = nn.Parameter(R)
        self.R = R
        self.r_encoder = CNN(in_channels, in_channels)
        # self.seg_cnn = SegCNN(in_channels, 1)
        # self.linear = nn.Linear(in_channels, in_channels)
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

    def forward(self, x_k, z, lz):
        B, C, Isize, Isize = x_k.shape
        _, zc, _, _ = z.shape
        # B, C, Isize_Isize = lz.shape
        # pdb.set_trace()

        # Zt
        # Zt = (self.R).T @ z.reshape(B, zc, -1)
        # Zt = Zt.reshape(Zt.shape[0], Zt.shape[1], Isize, Isize)
        # some bands have hri information and other are zero

        # LY = linear_operator.utils.sparse.bdsmm(lz.detach(), x_k.reshape(B, C, -1).transpose(1, 2))
        # LY = LY.reshape(B, C, Isize, Isize)
                
        ## melee idea: find all zeros channels in Zt create mask use mask to mix information
        # there are a lot of near zero values so this does not work
        # expanded_mask = (torch.sum(Zt, dim=(2, 3)) == 0)[:, :, None, None].repeat(1, 1, Isize, Isize)
        # pdb.set_trace()
        # Zt[expanded_mask] = LY[expanded_mask]        
        # learn a mask that makes sure low spectral information regions are selected
        # to minimize spectral distortion
        # soft_mask = self.seg_cnn(Zt)
        # pdb.set_trace()
        # s = torch.sum(Zt, dim=(2, 3))
        # soft_mask = self.linear(s/s.max())
        # pdb.set_trace()
        # soft_mask = s/s.max()
        # soft_mask = soft_mask[:, :, None, None].repeat(1, 1, Isize, Isize)
        # Zt = Zt*soft_mask + x_k*(1-soft_mask)
        # Zt = Zt + LY
        # x_k += self.r_encoder(z_aug)
        # z_aug = self.r_encoder(lz)
        z_aug = (self.R).T @ z.reshape(B, zc, -1)
        # pdb.set_trace()
        # x_new = x_new.reshape(x_new.shape[0], x_new.shape[1], -1)
       
        # z_aug = linear_operator.utils.sparse.bdsmm((self.R).T, lz.reshape(B, C, -1))
        # pdb.set_trace()
        # z_aug = F.softmax(z_aug, -1)
        z_aug = z_aug.reshape(B, C, Isize, Isize)
        x_k += self.r_encoder(z_aug)
        # x_k = x_k*z_aug #self.r_encoder(z_aug)
        # lz_xt = linear_operator.utils.sparse.bdsmm(lz, x_k.transpose(1,2))
        # pdb.set_trace()
        # x_k = self.r_encoder(torch.cat([x_k, z], 1))
        y_hat, layers = self.encoder(x_k) # downsample from 512 to 64
        x_hat = self.decoder(y_hat, layers) # 31 channels
        return y_hat, x_hat
        

    def calc_loss(self, x_new, y_hat, lz, x_old, y):
        loss = nn.MSELoss()
        recon_loss_y = loss(y_hat, y)
        recon_loss_x = loss(x_new, x_old)
        # SAM loss
        sam_loss = calc_sam_loss(x_new, x_old) #+ calc_sam_loss(y_hat, y)
        # sam_loss = calc_sam_loss(Zt, x_old)
        # sam_loss = torch.tensor([0.0]).to(recon_loss_y.device)
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
        


# class Upsampler(nn.Module):
#     def __init__(self, dec_channels):
#         super(Upsampler, self).__init__()
#         self.d_conv_1 = nn.Conv2d(dec_channels, dec_channels, 
#                                   kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.d_bn_1 = nn.BatchNorm2d(dec_channels)
#         self.d_conv_2 = nn.Conv2d(dec_channels, dec_channels, 
#                                   kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.d_bn_2 = nn.BatchNorm2d(dec_channels)
#         self.d_conv_3 = nn.Conv2d(dec_channels, dec_channels, 
#                                   kernel_size=(3, 3), stride=(1, 1), padding=1)
        
#         self.d_bn_3 = nn.BatchNorm2d(dec_channels)
        

#     def forward(self, x):
#         # h2
#         x = F.interpolate(x, scale_factor=2)
#         x = self.d_conv_1(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
#         x = self.d_bn_1(x)
#         o1 = x

#         # h3
#         x = F.interpolate(x, scale_factor=2)
#         x = self.d_conv_2(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
#         x = self.d_bn_2(x)
#         o2 = x

#         # h4
#         x = F.interpolate(x, scale_factor=2)
#         x = self.d_conv_3(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
#         x = self.d_bn_3(x)  

#         return x, [o2, o1]


# class Downsampler(nn.Module):
#     def __init__(self, in_channels=512) -> None:
#         super(Downsampler, self).__init__()
#         self.e_conv_1 = nn.Conv2d(in_channels, in_channels, 
#                                   kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.e_bn_1 = nn.BatchNorm2d(in_channels)
#         self.e_conv_2 = nn.Conv2d(in_channels*2, in_channels, 
#                                   kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.e_bn_2 = nn.BatchNorm2d(in_channels)
#         self.e_conv_3 = nn.Conv2d(in_channels*2, in_channels, 
#                                   kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.e_bn_3 = nn.BatchNorm2d(in_channels)
#         self.pool = nn.MaxPool2d(2)
        

#     def forward(self, x, layers):  
#         #h1
#         x = self.e_conv_1(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
#         x = self.e_bn_1(x)
#         x = self.pool(x)
#         x = torch.cat([x, layers[0]], 1)
#         # pdb.set_trace()
#         #h2
#         x = self.e_conv_2(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True)    
#         x = self.e_bn_2(x)     
#         x = self.pool(x)
#         x = torch.cat([x, layers[1]], 1)
#         #h3
#         x = self.e_conv_3(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
#         # x = self.tv_layer(x)
#         x = self.e_bn_3(x)
#         x = self.pool(x)
#         return x

# class AutoEncoderAmazing(nn.Module):
#     def __init__(self, in_channels, dec_channels, R):
#         super(AutoEncoderAmazing, self).__init__()
#         self.in_channels = in_channels
#         self.dec_channels = dec_channels
#         self.R = R
#         self.downsampler = Downsampler(in_channels)
#         self.upsampler = Upsampler(dec_channels)
#         self.seg_cnn = SegCNN(in_channels, in_channels)
#         # Reinitialize weights using He initialization
#         for m in self.modules():
#             if isinstance(m, torch.nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight.detach())
#                 m.bias.detach().zero_()
#             elif isinstance(m, torch.nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight.detach())
#                 m.bias.detach().zero_()
#             elif isinstance(m, torch.nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.detach())
#                 m.bias.detach().zero_()

#     def forward(self, y, z):
#         _, zc, Isize, Isize = z.shape
#         B, C, _, _ = y.shape
#         # Zt
#         Zt = (self.R).T @ z.reshape(B, zc, -1)
#         Zt = Zt.reshape(Zt.shape[0], Zt.shape[1], Isize, Isize)
#         x_hat, layers  = self.upsampler(y)
#         # soft_mask = self.seg_cnn(Zt)
#         # x_hat = Zt*soft_mask + x_hat*(1-soft_mask)
#         #x_hat = x_hat #+ Zt
#         # y_hat, o1, o2 = self.encoder(x_k) # downsample from 512 to 64
#         y_hat = self.downsampler(x_hat, layers) # 31 channels
#         return y_hat, x_hat
        

#     def calc_loss(self, x_hat, y_hat, lz, y):
#         loss = nn.MSELoss()
#         recon_loss_y = loss(y_hat, y)
#         # SAM loss
#         sam_loss = calc_sam_loss(y_hat, y)
#         ### Graph Laplacian Loss
#         x_hat = x_hat.reshape(x_hat.shape[0], x_hat.shape[1], -1)
#         lz_xt = linear_operator.utils.sparse.bdsmm(lz, x_hat.transpose(1,2))
#         GL = MatmulLinearOperator(x_hat, lz_xt)
#         factor = x_hat.shape[-1]
#         GL = torch.diagonal(to_dense(GL)/factor, dim1=-2, dim2=-1).sum() # trace
#         return sam_loss, recon_loss_y, GL


# x = torch.rand(31, 512, 512)
# y = torch.rand(31, 64, 64)
# z = torch.rand(512, 512, 3).numpy()
# model = AutoEncoder(31, 31)
# y_hat, x_hat = model(x[None, ...])
# lz = getLaplacian(z, z.shape[2])

# xhd = x_hat.reshape(x_hat.shape[0], x_hat.shape[1], -1)
# GL = torch.sum((torch.diagonal(xhd)**2) * torch.diagonal(lz))

class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampler, self).__init__()
        self.e_conv_1 = nn.Conv2d(in_channels, out_channels, 
                    kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(out_channels)
        self.tv_layer = GeneralTV2DLayer(lmbd_init=30,num_iter=10)
        
    
    def forward(self, zd, yiq):
        
        x = torch.cat([zd, yiq], 1)
        x = self.e_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.tv_layer(x)
        x = self.e_bn_1(x)
        return x

class Downsampler2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampler2, self).__init__()
        self.e_conv_1 = nn.Conv2d(in_channels, in_channels, 
                    kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(in_channels)
        self.e_conv_2 = nn.Conv2d(in_channels, out_channels, 
                    kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_2 = nn.BatchNorm2d(out_channels)
        self.e_conv_3 = nn.Conv2d(out_channels, out_channels, 
                    kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_3 = nn.BatchNorm2d(out_channels)
        self.tv_layer = GeneralTV2DLayer(lmbd_init=30,num_iter=10)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, zd, yiq):
        # x = torch.cat([zd, yiq], 1)
        x = zd
        x = self.e_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        # x = self.tv_layer(x)
        x = self.e_bn_1(x)
        x = self.pool(x)

        x = self.e_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_2(x)
        x = self.pool(x)

        x = self.e_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_3(x)
        x = self.pool(x)
        x = self.tv_layer(x)
        return x
        
# class Downsampler(nn.Module):
#     def __init__(self, in_channels=512) -> None:
#         super(Downsampler, self).__init__()
#         self.e_conv_1 = nn.Conv2d(in_channels, in_channels, 
#                                   kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.e_bn_1 = nn.BatchNorm2d(in_channels)
#         self.e_conv_2 = nn.Conv2d(in_channels*2, in_channels, 
#                                   kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.e_bn_2 = nn.BatchNorm2d(in_channels)
#         self.e_conv_3 = nn.Conv2d(in_channels*2, in_channels, 
#                                   kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.e_bn_3 = nn.BatchNorm2d(in_channels)
#         self.pool = nn.MaxPool2d(2)
        

#     def forward(self, x, layers):  
#         #h1
#         x = self.e_conv_1(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
#         x = self.e_bn_1(x)
#         x = self.pool(x)
#         x = torch.cat([x, layers[0]], 1)
#         # pdb.set_trace()
#         #h2
#         x = self.e_conv_2(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True)    
#         x = self.e_bn_2(x)     
#         x = self.pool(x)
#         x = torch.cat([x, layers[1]], 1)
#         #h3
#         x = self.e_conv_3(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
#         # x = self.tv_layer(x)
#         x = self.e_bn_3(x)
#         x = self.pool(x)
#         return x

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class FE(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.e_conv_1 = nn.Conv2d(in_channels, in_channels//2, 
                    kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(in_channels//2)
        self.e_conv_2 = nn.Conv2d(in_channels//2, in_channels//2, 
                    kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.e_bn_2 = nn.BatchNorm2d(in_channels//2)

    def forward(self, x):
        x = self.e_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_1(x)
        x = self.e_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_2(x)
        return x


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x, mode="train_test"):
        pdb.set_trace()
        x = x.reshape(x.shape[0], -1) # flatten out on image dimensions
        if mode != "train_y":
            x = self.feature_extractor(x)
        features = self.scale_to_bounds(x)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res
