import torch
import torch.nn as nn
import math
import pdb

class PositionalEmbeddingFactory:
    @staticmethod
    def get_embedding(method, max_position, d_model):
        if method == "sinusoidal":
            return SinusoidalPositionalEmbedding(d_model)
        elif method == "learned":
            return LearnedPositionalEmbedding(max_position, d_model)
        elif method == "relative":
            return RelativePositionalEmbedding(max_position, d_model)
        elif method == "rotary":
            return RotaryPositionalEmbedding(d_model)
        elif method == "fourier":
            return FourierPositionalEmbedding(2, d_model)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        assert d_model % 4 == 0, "Embedding dimension must be divisible by 4 for 2D input."

    def forward(self, positions):
        N = positions.shape[0]
        embeddings = torch.zeros((N, self.d_model), dtype=torch.float32, device=positions.device)
        d_model_half = self.d_model // 2
        div_term = torch.exp(torch.arange(0, d_model_half, 2, device=positions.device) * -(math.log(10000.0) / d_model_half))
        
        for i, dim in enumerate(['x', 'y']):
            pos = positions[:, i].unsqueeze(1)
            embeddings[:, i*d_model_half:(i+1)*d_model_half:2] = torch.sin(pos * div_term)
            embeddings[:, i*d_model_half+1:(i+1)*d_model_half:2] = torch.cos(pos * div_term)
        
        return embeddings

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_position, d_model):
        super().__init__()
        self.embedding_x = nn.Embedding(max_position, d_model // 2)
        self.embedding_y = nn.Embedding(max_position, d_model // 2)

    def forward(self, positions):
        x_emb = self.embedding_x(positions[:, 0].long())
        y_emb = self.embedding_y(positions[:, 1].long())
        return torch.cat([x_emb, y_emb], dim=-1)

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, max_position, d_model):
        super().__init__()
        self.max_position = max_position
        self.embedding = nn.Embedding(2 * max_position + 1, d_model // 2)

    def forward(self, positions):
        N = positions.shape[0]
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions_clipped = torch.clamp(relative_positions, -self.max_position, self.max_position)
        relative_positions_shifted = relative_positions_clipped + self.max_position
        
        emb_x = self.embedding(relative_positions_shifted[:, :, 0].long())
        emb_y = self.embedding(relative_positions_shifted[:, :, 1].long())
        
        return torch.cat([emb_x, emb_y], dim=-1)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        assert d_model % 4 == 0, "Embedding dimension must be divisible by 4 for 2D input."

    def forward(self, positions):
        N = positions.shape[0]
        d_model_quarter = self.d_model // 4
        inv_freq = 1. / (10000 ** (torch.arange(0, d_model_quarter, device=positions.device).float() / d_model_quarter))
        
        embeddings = torch.zeros((N, self.d_model), dtype=torch.float32, device=positions.device)
        for i, dim in enumerate(['x', 'y']):
            pos = positions[:, i].unsqueeze(1)
            sinusoid_inp = torch.einsum('i,j->ij', pos, inv_freq)
            emb_sin, emb_cos = torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
            
            start = i * self.d_model // 2
            end = (i + 1) * self.d_model // 2
            embeddings[:, start:end:2] = emb_sin
            embeddings[:, start+1:end:2] = emb_cos
        
        return embeddings


class FourierPositionalEmbedding(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, positions):
        f = 2 * math.pi * positions @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

# Usage example:
if __name__ == "__main__":
    max_position = 1000  # Maximum position value
    d_model = 256  # Embedding dimension
    N = 100  # Number of positions

    # Generate random 2D positions
    positions = torch.rand(N, 2) * max_position

    # Try different embedding methods
    methods = ["sinusoidal", "learned", "fourier"]

    for method in methods:
        embedding_layer = PositionalEmbeddingFactory.get_embedding(method, max_position, d_model)
        embeddings = embedding_layer(positions)
        print(f"{method.capitalize()} Embedding shape:", embeddings.shape)