import torch.nn as nn
import torch.nn.functional as F
from .unet_base_blocks import Conv1x3x1
from .utils import pad_to_power_of_2
import torch
import pdb


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
        # self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(in_channels)
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

    def forward(self, x, skip_connection):
        
        # x = self.bn(F.relu(self.conv(z)))
        x = self.bn3(F.relu(self.deconv3(x)))
        x = torch.cat((x, skip_connection[1]), dim=1) 
        x = self.bn2(F.relu(self.deconv2(x)))
        x = torch.cat((x, skip_connection[0]), dim=1)
        x = self.bn1(F.relu(self.deconv1(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, msi_in, latent_dim):
        super().__init__()
        # msi_enc -> 3 x H x W -> -> 256  x 1 x 1
        self.msi_enc = Down1x3x1(msi_in, latent_dim)
        
    def forward(self, msi):
        z_msi, msi_out = self.msi_enc(msi) # apply bilinear upsample here
        return z_msi, msi_out


class SegmentationDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.decoder = Up(latent_dim, out_channels)

    def forward(self, z, msi_out):
        x = self.decoder(z, msi_out)
        return x

class UNet(nn.Module):
    def __init__(self, msi_in, latent_dim, output_channels, **kwargs):
        super().__init__()
        self.encoder = Encoder(msi_in, latent_dim)
        self.decoder = SegmentationDecoder(latent_dim, output_channels)
        
    def forward(self, hsi=None, msi=None):
        orig_ht, orig_width = msi.shape[2:]
        msi = pad_to_power_of_2(msi)
        z_msi, msi_out = self.encoder(msi)
        segmentation_map = self.decoder(z_msi, msi_out)
        preds = segmentation_map[:, :, :orig_ht, :orig_width]
        return {
            'preds': preds
        }


if __name__ == '__main__':
    # usage
    model = UNet(3, 256, 5)  # Assume output channels for segmentation map is 5
    for i in range(1, 5):
        msi = torch.rand(2, 3, 256*i, 256*i)
        output = model(msi=msi)
        print(output.shape)  

