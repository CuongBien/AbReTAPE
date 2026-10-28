# -*- coding: utf-8 -*-
# Mạng Decoder và Transpose-Lift Decoder tấn công tái tạo ảnh từ biểu diễn trung gian z (IR)
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder tái tạo ảnh x_hat từ biểu diễn trung gian IR z.
    - Với CIFAR (ảnh 32x32, upscale_factor=1):
      Đầu vào: z in R^(B x in_channels x 32 x 32) -> Đầu ra: x_hat in R^(B x out_channels x 32 x 32)
    - Với PCAM (ảnh 64x64, upscale_factor=2):
      Đầu vào: z in R^(B x in_channels x 32 x 32) -> Upsample 2x -> Đầu ra: x_hat in R^(B x out_channels x 64 x 64)
    """
    def __init__(self, in_channels=64, out_channels=3, hidden_dim=128, upscale_factor=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if upscale_factor > 1:
            layers.extend([
                nn.Upsample(scale_factor=upscale_factor, mode="bilinear", align_corners=False),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(64, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.upscale_factor = upscale_factor

    def forward(self, z):
        return self.net(z)


class TransposeLiftConvDecoder(nn.Module):
    """
    Attacker Conv cho biểu diễn zt:
    - Nếu zt đã ở không gian cut-layer D=65536 (vd: B0), chuyển về (B, 64, 32, 32) rồi qua ConvDecoder.
    - Nếu zt là biểu diễn nén k (vd: B7 LightSplit k=2048), dùng ma trận chuyển vị W_lift = R^T
      để nâng lên D=65536, chuyển về (B, 64, 32, 32) rồi qua ConvDecoder.
    - Hỗ trợ upscale_factor (1 cho CIFAR 32x32, 2 cho PCAM 64x64).
    """
    def __init__(self, bottleneck_weight=None, in_channels=64, out_channels=3, upscale_factor=1):
        super().__init__()
        if bottleneck_weight is not None:
            # bottleneck_weight có shape [k, D] -> W_lift là [D, k]
            self.register_buffer("W_lift", bottleneck_weight.t().clone())
        else:
            self.W_lift = None
        self.conv_decoder = Decoder(
            in_channels=in_channels,
            out_channels=out_channels,
            upscale_factor=upscale_factor
        )
        self.upscale_factor = upscale_factor

    def forward(self, zt):
        if self.W_lift is not None:
            if zt.dim() > 2:
                zt = zt.view(zt.size(0), -1)
            # zt: [B, k] @ W_lift.t() [k, D] -> [B, D]
            zh = torch.matmul(zt, self.W_lift.t())
            zh_tensor = zh.view(zt.size(0), 64, 32, 32)
        else:
            if zt.dim() == 2:
                zh_tensor = zt.view(zt.size(0), 64, 32, 32)
            else:
                zh_tensor = zt
        return self.conv_decoder(zh_tensor)
