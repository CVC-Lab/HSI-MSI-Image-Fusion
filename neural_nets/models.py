import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb


class Down1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.l2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.l3 = nn.Conv2d(128, out_channels, kernel_size=1, stride=2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP layer
        
    def forward(self, x):
        x1 = self.bn1(F.relu(self.l1(x)))
        x2 = self.bn2(F.relu(self.l2(x1)))
        x3 = self.bn3(F.relu(self.l3(x2)))
        z = self.gap(x3)
        return z, [x1, x2, x3]


class UpConcat(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        # upsample by factor 4
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, 
                                         kernel_size=4, stride=4, 
                                         padding=0, output_padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x, y):
        x_up = self.deconv(x)
        return self.bn(F.relu(x_up+y))
        
        
class Up1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # upsample latent to match spatial extent
        self.up3 = nn.ConvTranspose2d(in_channels*2, in_channels, 
                                         kernel_size=4, stride=4, 
                                         padding=0, output_padding=0)
        self.bup3 = nn.BatchNorm2d(in_channels)
        self.up2 = nn.ConvTranspose2d(in_channels, in_channels, 
                                         kernel_size=4, stride=4, 
                                         padding=0, output_padding=0)
        self.bup2 = nn.BatchNorm2d(in_channels)
        self.up1 = nn.ConvTranspose2d(in_channels, in_channels, 
                                         kernel_size=2, stride=2, 
                                         padding=0, output_padding=0)
        self.bup1 = nn.BatchNorm2d(in_channels)
        
        # up 1x3x1
        self.deconv3 = nn.ConvTranspose2d(in_channels*2, 128, 
                                          kernel_size=1, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128*2, 64, kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64*2, out_channels, 
                                          kernel_size=1, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, z, skip_connection):
        print(z.shape)
        z = self.bup3(F.relu(self.up3(z)))
        z = self.bup2(F.relu(self.up2(z)))
        x = self.bup1(F.relu(self.up1(z)))
        print(x.shape)
        
        x = torch.cat((x, skip_connection[2]), dim=1)
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
        self.hsi_enc = Down1x3x1(hsi_in, latent_dim)
        # msi_enc -> 3 x H x W -> -> 256  x 1 x 1
        self.msi_enc = Down1x3x1(msi_in, latent_dim)
        
    def forward(self, hsi, msi):
        z_hsi, hsi_out = self.hsi_enc(hsi)
        z_msi, msi_out = self.msi_enc(msi)
        print(z_hsi.shape, z_msi.shape, hsi_out[0].shape, msi_out[0].shape)
        return z_hsi, z_msi, hsi_out, msi_out


class SegmentationDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.upcat3 = UpConcat(latent_dim)# [B, 256, 32, 32]
        self.upcat2 = UpConcat(latent_dim//2)# [B, 128, 64, 64]
        self.upcat1 = UpConcat(latent_dim//4)# [B, 64, 128, 128]
        self.decoder = Up1x3x1(latent_dim, out_channels)

    def forward(self, z, hsi_out, msi_out):
        # merge outputs of hsi and msi encoder
        out3 = self.upcat3(msi_out[2], hsi_out[2])
        out2 = self.upcat2(msi_out[1], hsi_out[1])
        out1 = self.upcat1(msi_out[0], hsi_out[0])
        x = self.decoder(z, [out1, out2, out3])
        return x

class SiameseUNet(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim, output_channels):
        super().__init__()
        self.encoder = SiameseEncoder(hsi_in, msi_in, latent_dim)
        self.decoder = SegmentationDecoder(latent_dim, output_channels)
        
    def forward(self, hsi, msi):
        z_hsi, z_msi, hsi_out, msi_out = self.encoder(hsi, msi)
        z = torch.cat([z_hsi, z_msi], dim=1)
        segmentation_map = self.decoder(z, hsi_out, msi_out)
        segmentation_map = segmentation_map.softmax(dim=1)
        return segmentation_map


if __name__ == '__main__':
    # usage
    hsi = torch.rand(2, 31, 256, 256)
    msi = torch.rand(2, 3, 64, 64)
    model = SiameseUNet(31, 3, 256, 5)  # Assume output channels for segmentation map is 5
    output = model(hsi, msi)
    print(output.shape)  # Should be [2, 5, 256, 256]
