import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from einops import rearrange
from utils import pad_to_power_of_2
import pdb

class ViTEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim, config):
        super(ViTEncoder, self).__init__()
        self.config = config
        self.patch_size = config["patch_size"]
        assert latent_dim % 12 == 0
        # ViT Configuration
        vit_config = ViTConfig(
            hidden_size=latent_dim,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            patch_size=self.patch_size if self.patch_size else 1,
            num_channels=input_channels,
            image_size=config["input_size"],  # Assuming a square input
        )
        
        self.vit = ViTModel(vit_config)
    
    def forward(self, x):
        pdb.set_trace()
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state

class SiameseViTEncoder(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim, config):
        super().__init__()
        self.hsi_vit = ViTEncoder(hsi_in, latent_dim, config["hsi"])
        self.msi_vit = ViTEncoder(msi_in, latent_dim, config["msi"])
        
    def forward(self, hsi, msi):
        z_hsi = self.hsi_vit(hsi)
        z_msi = self.msi_vit(msi)
        return z_hsi, z_msi

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

class SiameseViTUNet(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim, output_channels, config):
        super().__init__()
        self.encoder = SiameseViTEncoder(hsi_in, msi_in, latent_dim, config)
        self.decoder = SegmentationDecoder(latent_dim, output_channels)
        
    def forward(self, hsi, msi):
        orig_ht, orig_width = msi.shape[2:]
        hsi = hsi.to(torch.double)
        msi = msi.to(torch.double)
        msi = pad_to_power_of_2(msi)
        hsi = pad_to_power_of_2(hsi)
        z_hsi, z_msi = self.encoder(hsi, msi)
        z = torch.cat([z_hsi, z_msi], dim=1)
        pdb.set_trace()
        # segmentation_map = self.decoder(z, hsi_out, msi_out)    
        # return segmentation_map[:, :, :orig_ht, :orig_width]

if __name__ == '__main__':
    # usage
    for i in range(1, 5):
        config = {
            "hsi":{
                "patch_size": 8,
                "input_size": 64*i
            },
            "msi":{
                "patch_size": 8,
                "input_size": 256*i
            }
        }
        model = SiameseViTUNet(31, 3, 252, 5, config)  # Assume output channels for segmentation map is 5
        hsi = torch.rand(2, 31, 64*i, 64*i)
        msi = torch.rand(2, 3, 256*i, 256*i)
        output = model(hsi, msi)
        print(output.shape)