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
import torch.nn.functional as F
import pdb
import scipy
dtype = torch.FloatTensor
from torchmetrics import SpectralAngleMapper

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
