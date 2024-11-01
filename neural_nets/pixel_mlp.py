import torch
import torch.nn as nn
import torch.nn.functional as F
from.utils import activation_layers
from einops import rearrange
import pdb
from .position_embeddings import PositionalEmbeddingFactory

class FourierFeatureEmbedding(nn.Module):
    def __init__(self, embedding_dim, input_dim, a):
        super().__init__()
        self.b = torch.rand(embedding_dim, input_dim).cuda().to(torch.double)
        self.a = a
    
    def forward(self, x):
        return torch.cat([
            self.a*torch.sin(2*torch.pi*x) @ self.b.T, 
            self.a*torch.cos(2*torch.pi*x) @ self.b.T], axis=-1)


feature_embedding = {
    'fourier': FourierFeatureEmbedding
}

class PixelMLP(nn.Module):
    def __init__(self, hsi_in, msi_in, 
                 feature_embedding_dim, position_embedding_dim, a,
                 output_channels, act, fe_alg, pe_alg, input_mode, **kwargs):
        super().__init__()
        self.position_embedding_dim = position_embedding_dim
        self.pe = PositionalEmbeddingFactory.get_embedding(pe_alg, 
                                                           2048, # not relevant rn
                                                           position_embedding_dim)
        if input_mode == 'hsi_only':
            fdim = hsi_in
        elif input_mode == 'msi_only':
            fdim = msi_in
        else:
            fdim = hsi_in + msi_in
        self.fe_alg = fe_alg
        if fe_alg == 'fourier':   
            self.fe = feature_embedding[fe_alg](feature_embedding_dim, 
                                            fdim, a)  
            fdim = feature_embedding_dim*2 
        self.hsi_in = hsi_in
        self.msi_in = msi_in
        self.input_mode = input_mode
        total_in = fdim + position_embedding_dim
        self.net = nn.Sequential(*[
            nn.Linear(total_in, total_in),
            activation_layers[act](),
            nn.Linear(total_in, (total_in)//2),
            activation_layers[act](),
            nn.Linear((total_in)//2, (total_in)//4),
            activation_layers[act](),
            nn.Linear((total_in)//4, (total_in)//4),
            activation_layers[act](),
            nn.Linear((total_in)//4, (total_in)//8),
            activation_layers[act](),
            nn.Linear((total_in)//8, output_channels),
        ])
    
    def forward(self, x, y=None):
        patch_type_data = False
        if len(x.shape) == 3:
            B, N, C = x.shape
            x = rearrange(x, "B N C -> (B N) C")
            patch_type_data = True
            
        position_values = x[:,  -2:]
        pos_emb = self.pe(position_values)
        # x - [rgb_pixel, hsi_super_pixel, position_values]
        if self.input_mode == 'hsi_only':
            pixel_values = x[:, 3:-2]
        elif self.input_mode == 'msi_only':
            pixel_values = x[:, :3]
        else:
            pixel_values = x[:, :-2]
        if self.fe_alg == 'fourier':
            feat_emb = self.fe(pixel_values)
        else:
            feat_emb = pixel_values
        x = torch.cat([feat_emb, pos_emb], dim=-1)
        x = self.net(x) # add softmax as part of loss func
        if patch_type_data:
            H, W = y.shape[1:3]
            x = x.reshape(B, H, W, -1)
        return {
            'preds': x
        }
    
if __name__ == '__main__':
    
    B, C, H, W = 2, 12, 50, 50
    embedding_dim = 256
    a = 1.0
    x = torch.rand(2, C, H, W)
    # Step 1: Generate the pixel coordinates
    # Generate a grid of coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(H), 
                                        torch.arange(W), indexing='ij')
    y_coords = y_coords.unsqueeze(0).repeat(B, 1, 1)  # Repeat for batch size
    x_coords = x_coords.unsqueeze(0).repeat(B, 1, 1)  # Repeat for batch size
    # Step 2: Flatten the spatial dimensions
    # Reshape x to have shape (B, C, H * W)
    x_flattened = x.view(B, C, -1)  # Flatten H and W
    x_coords_flattened = x_coords.view(B, -1)  # Flatten coordinates
    y_coords_flattened = y_coords.view(B, -1)
    # Step 3: Concatenate coordinates and values
    # Stack coordinates to form (B, 2, H * W)
    pixel_coordinates = torch.stack([x_coords_flattened, y_coords_flattened], dim=1)
    # Stack the coordinates and pixel values along the third dimension
    # Resulting shape: (B, H * W, 2 + C)
    result = torch.cat([pixel_coordinates.permute(0, 2, 1), x_flattened.permute(0, 2, 1)], dim=2)
    # Result tensor will have shape (B, num_pixels, pixel_info)
    # Where pixel_info = 2 (for coordinates) + C (for pixel values)
    print(result.shape)  # Should be (2, 50*50, 2+3)
    model = PixelMLP(input_dim=C, 
                          out_dim=6, 
                          embedding_dim=embedding_dim,
                          a=a, 
                          act='snake', pe='fourier')
    out = model(result)
    print(out.shape)
    # so, coordinates for RGB image must be 0-1 and then we must 
    # somehow index the coordinates of the HSI to align 
    # with coordinates in RGB. 