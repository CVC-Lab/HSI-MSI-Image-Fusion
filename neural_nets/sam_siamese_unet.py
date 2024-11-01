import torch.nn as nn
import torch.nn.functional as F
import torch
from segment_anything import sam_model_registry, SamPredictor
from einops import pack
from .unet_base_blocks import Conv1x3x1
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
        x = torch.cat((x, skip_connection[1]), dim=1) # 
        x = self.bn2(F.relu(self.deconv2(x)))
        x = torch.cat((x, skip_connection[0]), dim=1) # torch.Size([2, 64, 64, 64])
        x = self.bn1(F.relu(self.deconv1(x)))
        return x

class ScaleCatConv(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, in_dim//2, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_dim//2)
    def forward(self, big, small):
        sx, sy = big.shape[-2] / small.shape[-2], big.shape[-1] / small.shape[-1]
        small = F.interpolate(small, scale_factor=(sx, sy))
        out, _ = pack([big, small], "B * H W")
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

class MSIEncoder(nn.Module):
    def __init__(self, 
                 msi_in,
                 latent_dim,
                 checkpoint="/workspace/shubham/HSI-MSI-Image-Fusion/models/sam_vit_h_4b8939.pth"):
        model_type = "vit_h"
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to("cuda")
        self.predictor = SamPredictor(sam)
        self.l1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.l2 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.msi_enc = Down1x3x1(msi_in, latent_dim)
        # cat outputs of l1, l2
        self.z1catconv = ScaleCatConv(64*2) 
        self.z2catconv = ScaleCatConv(128*2)
        self.zcatconv = ScaleCatConv(256*2)
        
        
    
    def forward(self, msi):
        z_sam_msi = []
        B = msi.shape[0]
        for i in range(B):
            self.predictor.set_image(msi[i])
            z_sam_msi.append(self.predictor.get_image_embedding())
        
        z3_sam, _ = pack(z_sam_msi, "* C H W")
        z3_sam = z3_sam.double()
        z1_sam = self.bn1(F.relu(self.l1(z3_sam)))
        z2_sam = self.bn2(F.relu(self.l2(z3_sam)))
        z3, z_out = self.msi_enc(msi)
        
        z_msi = self.zcatconv(z3, z3_sam)
        z1 = self.z1catconv(z_out[0], z1_sam)
        z2 = self.z2catconv(z_out[1], z2_sam)
        return z_msi, [z1, z2]
        
        


class SiameseEncoder(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim):
        super().__init__()
        # hsi_enc -> 31 x h x w -> 256  x 1 x 1
        self.hsi_enc = Down1x3x1(hsi_in, latent_dim)
        # msi_enc -> 3 x H x W -> -> 256  x 1 x 1
        # self.msi_enc = Down1x3x1(msi_in, latent_dim)
        self.msi_enc = MSIEncoder(msi_in, latent_dim)
        
    def forward(self, hsi, msi):
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

class SamSiameseUNet(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim, output_channels):
        super().__init__()
        self.encoder = SiameseEncoder(hsi_in, msi_in, latent_dim)
        self.decoder = SegmentationDecoder(latent_dim, output_channels)
        
    def forward(self, hsi, msi):
        z_hsi, z_msi, hsi_out, msi_out = self.encoder(hsi, msi)
        z = torch.cat([z_hsi, z_msi], dim=1)
        segmentation_map = self.decoder(z, hsi_out, msi_out)
        return segmentation_map


if __name__ == '__main__':
    # usage
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamSiameseUNet(31, 3, 256, 5)  # Assume output channels for segmentation map is 5
    model.to(DEVICE).to(torch.double)
    for i in range(1, 5):
        hsi = torch.rand(2, 31, 64*i, 64*i).to(torch.double).to(DEVICE)
        msi = torch.rand(2, 3, 256*i, 256*i).to(torch.double).to(DEVICE)
        output = model(hsi, msi)
        print(output.shape)  

