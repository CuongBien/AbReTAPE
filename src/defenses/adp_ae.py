# Baseline B6: ADP-style Autoencoder Defense (Plug-in Feature Map Defense, arXiv:2502.20629)
import torch
import torch.nn as nn


class PerturbAE(nn.Module):
    """
    B6 — AE plug-in kiểu ADP (arXiv:2502.20629) tại cut layer.
    AE sinh nhiễu loạn đối kháng delta(z), kẹp biên độ bằng alpha * tanh.
    Kiến trúc nén và tái tạo:
        Encoder: Conv(64 -> 32) -> ReLU -> Conv(32 -> 16) -> ReLU
        Decoder: Conv(16 -> 32) -> ReLU -> Conv(32 -> 64)
    """
    def __init__(self, channels=64, alpha=0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.enc = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        return self.alpha * torch.tanh(self.dec(self.enc(z)))


class ADPAutoEncoderDefense(nn.Module):
    """
    Wrapper tích hợp PerturbAE vào pipeline Split Learning chuẩn:
        z' = z + delta(z) = z + PerturbAE(z)
    """
    def __init__(self, in_channels=64, alpha=0.1, **kwargs):
        super().__init__()
        self.perturb_ae = PerturbAE(channels=in_channels, alpha=alpha)

    def forward(self, z):
        return z + self.perturb_ae(z)
