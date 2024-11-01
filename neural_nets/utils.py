import torch
import torch.nn as nn
import torch.nn.functional as F

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def pad_to_power_of_2(x):
    _, _, h, w = x.size()
    new_h = next_power_of_2(h)
    new_w = next_power_of_2(w)
    pad_h = new_h - h
    pad_w = new_w - w
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    x_padded = F.pad(x, padding, "constant", 0).to(x.dtype).to(x.device)
    return x_padded

class Snake(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2


activation_layers = {
    'relu': nn.ReLU,
    'snake': Snake,
    'gelu': nn.GELU
}