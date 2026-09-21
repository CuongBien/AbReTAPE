# Baseline B1: Thêm nhiễu Gaussian độc lập tại cut layer
import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """
    Thêm nhiễu Gauss độc lập tại cut layer:
        z' = z + sigma * epsilon, với epsilon ~ N(0, I)
    
    Khi lan truyền ngược:
        dL/dz = dL/dz' * dz'/dz = dL/dz' * 1 = dL/dz'
    Gradient được truyền thẳng qua nhiễu về phía Client.
    """
    def __init__(self, sigma=0.5, apply_on_eval=True):
        super().__init__()
        self.sigma = float(sigma)
        self.apply_on_eval = apply_on_eval

    def forward(self, z):
        if self.sigma <= 0.0:
            return z
        if self.training or self.apply_on_eval:
            noise = torch.randn_like(z) * self.sigma
            return z + noise
        return z

    def extra_repr(self):
        return f"sigma={self.sigma}, apply_on_eval={self.apply_on_eval}"
