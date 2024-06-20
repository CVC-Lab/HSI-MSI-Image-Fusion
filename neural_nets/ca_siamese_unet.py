import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
from .unet_base_blocks import Conv1x3x1
from .utils import pad_to_power_of_2
## Siamese Unet with Learnable Channel attention

# Importance Weighted Channel Attention
class IWCA(nn.Module):
    def __init__(self, in_channels):
        super(IWCA, self).__init__()
        # no mixing of channel information
        self.c0 = nn.Conv2d(in_channels, in_channels, 
                                    kernel_size=3, groups=in_channels, padding=1)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.c1 = nn.Conv2d(in_channels, in_channels, 
                                    kernel_size=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # # mixing of channel information
        self.c2 = nn.Conv2d(in_channels, in_channels, 
                                    kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.c3 = nn.Conv2d(in_channels, in_channels, 
                                    kernel_size=1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.importance_wts = None

    def forward(self, x):
        # Group Convolution
        x_c = self.bn0(F.relu(self.c0(x)))
        x_c = self.bn1(F.relu(self.c1(x_c)))
        # Global Average Pooling
        avg_out = self.global_avg_pool(x_c)
        # Apply sigmoid to get the importance weights
        importance_weights = self.sigmoid(avg_out)
        self.importance_wts = importance_weights
        x_s = self.bn2(F.relu(self.c2(x)))
        x_s = self.bn3(F.relu(self.c3(x_s)))
        # Scale the original input
        out = x * importance_weights
        return out


class Down1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = Conv1x3x1(in_channels, 64)
        self.l2 = Conv1x3x1(64, 128)
        self.l3 = Conv1x3x1(128, out_channels)
        
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x3, [x1, x2]
    
class UpConcat(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, msi_feat, hsi_feat):
        # upsample msi features
        sx, sy = msi_feat.shape[-2] // hsi_feat.shape[-2], msi_feat.shape[-1] // hsi_feat.shape[-1]
        hsi_feat = F.interpolate(hsi_feat, scale_factor=(sx, sy))
        out = torch.cat([hsi_feat, msi_feat], dim=1)
        return self.bn(F.relu(self.conv(out)))
        
        
        
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # upsample latent to match spatial extent
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.deconv3 = nn.ConvTranspose2d(in_channels, 128, 
                                          kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128*2, 64, kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64*2, out_channels, 
                                          kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, z, skip_connection):
        
        x = self.bn(F.relu(self.conv(z)))
        x = self.bn3(F.relu(self.deconv3(x)))
        x = torch.cat((x, skip_connection[1]), dim=1) 
        x = self.bn2(F.relu(self.deconv2(x)))
        x = torch.cat((x, skip_connection[0]), dim=1)
        x = self.bn1(F.relu(self.deconv1(x)))
        return x
    
            
class Up1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # upsample latent to match spatial extent
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        # up 1x3x1
        self.deconv3 = nn.ConvTranspose2d(in_channels, 128, 
                                          kernel_size=1, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128*2, 64, kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64*2, out_channels, 
                                          kernel_size=1, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, z, skip_connection):
        x = self.bn(F.relu(self.conv(z)))
        x = self.bn3(F.relu(self.deconv3(x)))
        x = torch.cat((x, skip_connection[1]), dim=1) 
        x = self.bn2(F.relu(self.deconv2(x)))
        x = torch.cat((x, skip_connection[0]), dim=1)
        x = self.bn1(F.relu(self.deconv1(x)))
        return x


class SiameseEncoder(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim):
        super().__init__()
        # hsi_enc -> 31 x h x w -> 256  x 1 x 1
        self.channel_selector = IWCA(hsi_in)
        self.hsi_enc = Down1x3x1(hsi_in, latent_dim)
        # msi_enc -> 3 x H x W -> -> 256  x 1 x 1
        self.msi_enc = Down1x3x1(msi_in, latent_dim)
        
    def forward(self, hsi, msi):
        hsi = self.channel_selector(hsi)
        z_hsi, hsi_out = self.hsi_enc(hsi)
        z_msi, msi_out = self.msi_enc(msi) # apply bilinear upsample here
        # get scale of upsampling
        sx, sy = z_msi.shape[-2] // z_hsi.shape[-2], z_msi.shape[-1] // z_hsi.shape[-1]
        z_hsi = F.interpolate(z_hsi, scale_factor=(sx, sy))
        return z_hsi, z_msi, hsi_out, msi_out


class SegmentationDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.upcat2 = UpConcat(latent_dim//2)# [B, 128, 64, 64]
        self.upcat1 = UpConcat(latent_dim//4)# [B, 64, 128, 128]
        self.decoder = Up(latent_dim, out_channels)

    def forward(self, z, hsi_out, msi_out):
        # merge outputs of hsi and msi encoder
        out2 = self.upcat2(msi_out[1], hsi_out[1])
        out1 = self.upcat1(msi_out[0], hsi_out[0])
        x = self.decoder(z, [out1, out2])
        return x

class CASiameseUNet(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim, output_channels):
        super().__init__()
        self.encoder = SiameseEncoder(hsi_in, msi_in, latent_dim)
        self.decoder = SegmentationDecoder(latent_dim, output_channels)
        
    def forward(self, hsi, msi):
        orig_ht, orig_width = msi.shape[2:]
        hsi = hsi.to(torch.double)
        msi = msi.to(torch.double)
        msi = pad_to_power_of_2(msi)
        hsi = pad_to_power_of_2(hsi)
        
        z_hsi, z_msi, hsi_out, msi_out = self.encoder(hsi, msi)
        z = torch.cat([z_hsi, z_msi], dim=1)
        segmentation_map = self.decoder(z, hsi_out, msi_out)    
        return segmentation_map[:, :, :orig_ht, :orig_width]


if __name__ == '__main__':
    # usage
    model = CASiameseUNet(31, 3, 256, 5)  # Assume output channels for segmentation map is 5
    for i in range(1, 5):
        hsi = torch.rand(2, 31, 64*i, 64*i)
        msi = torch.rand(2, 3, 256*i, 256*i)
        output = model(hsi, msi)
        print(output.shape)
        # instead of output, we will use the loss to compute which channel 
        # inflences the training more than others
        # lower loss also means that those channels are better
        # full jacobian -> [2, 5, 256, 256, 2, 31, 1, 1] so take mean 
        # jacobian computation
        # output = model(hsi, msi)
        # jacobian = torch.autograd.functional.jacobian(lambda x: output.mean((2, 3)), 
        #                                 model.encoder.channel_selector.importance_wts)
        # jacobian = jacobian.squeeze() # [2, 5, 2, 31]
        # jacobian = jacobian.mean((0, 2)) # [5, 31]
        # print(jacobian.shape)
        

