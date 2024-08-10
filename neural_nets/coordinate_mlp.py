import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import activation_layers
from einops import rearrange


class FourierPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, a):
        super().__init__()
        self.b = torch.rand(embedding_dim, 2)#\
                    #.to(torch.double)#.to(torch.cuda)
        self.a = a
    
    def forward(self, x):
        return torch.cat([
            self.a*torch.sin(2*torch.pi*x) @ self.b.T, 
            self.a*torch.cos(2*torch.pi*x) @ self.b.T], axis=-1)


positional_embedding = {
    'fourier': FourierPositionalEmbedding
}

class CoordinateMLP(nn.Module):
    def __init__(self, input_dim, embedding_dim, a,
                 out_dim, act, pe):
        super().__init__()
        self.pe = positional_embedding[pe](embedding_dim, a)
        if act == 'fourier':
            out_emb_dim = (embedding_dim*2)
        else: 
            out_emb_dim = embedding_dim
        self.net = nn.Sequential(*[
            nn.Linear(input_dim+out_emb_dim, (input_dim+out_emb_dim)*2),
            activation_layers[act](),
            nn.Linear((input_dim+out_emb_dim)*2, (input_dim+out_emb_dim)//2),
            activation_layers[act](),
            nn.Linear((input_dim+out_emb_dim)//2, (input_dim+out_emb_dim)//4),
            activation_layers[act](),
            nn.Linear((input_dim+out_emb_dim)//4, out_dim),
        ])
    
    def forward(self, x):
        pixel_values, position_values = x[:, :, :-2], x[:, :, -2:]
        pos_emb = self.pe(position_values)
        x = torch.cat([pixel_values, pos_emb], dim=-1)
        x = self.net(x) # add softmax as part of loss func
        return x
    
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
    model = CoordinateMLP(input_dim=C, 
                          out_dim=6, 
                          embedding_dim=embedding_dim,
                          a=a, 
                          act='snake', pe='fourier')
    out = model(result)
    print(out.shape)
    # so, coordinates for RGB image must be 0-1 and then we must 
    # somehow index the coordinates of the HSI to align 
    # with coordinates in RGB. 