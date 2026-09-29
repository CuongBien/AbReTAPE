# Baseline B6: ADP-style Autoencoder Defense (Plug-in Feature Map Defense, arXiv:2502.20629)
import torch
import torch.nn as nn


class ADPAutoEncoderDefense(nn.Module):
    """
    Baseline B6 (ADP - Adversarial Distortion Plug-in):
    Sử dụng kiến trúc Autoencoder nút thắt cổ chai (bottleneck) tại cut-layer để nén và tái tạo
    biểu diễn đặc trưng, kết hợp với phép nhiễu loạn phân phối nhằm làm mờ cấu trúc cục bộ.
    """
    def __init__(self, in_channels=64, bottleneck_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, z):
        # Nén qua bottleneck rồi giải mã
        latent = self.encoder(z)
        z_out = self.decoder(latent) * z
        return z_out
