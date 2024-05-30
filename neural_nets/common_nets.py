import torch
import torch.nn as nn
import torch.nn.functional as F


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
        

'''
import linear_operator
from linear_operator.operators import MatmulLinearOperator
from linear_operator import to_dense
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

'''


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