# Bước 1 — Decoder tấn công tái tạo bị động (Passive Reconstruction Attack)
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder tái tạo ảnh x_hat từ biểu diễn trung gian IR z.
    Đầu vào: z in R^(B x 64 x 32 x 32)
    Đầu ra:  x_hat in R^(B x 3 x 32 x 32)
    """
    def __init__(self, in_channels=64, out_channels=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        return self.net(z)
