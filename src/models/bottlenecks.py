# -*- coding: utf-8 -*-
"""
src/models/bottlenecks.py
Định nghĩa kiến trúc Client và Server kết hợp Nút thắt cổ chai (Bottleneck)
và In-training Adversary (LearnedAdaptiveDecoder) cho Split Learning.
Hỗ trợ cả Fixed Orthogonal Projection (Mode F) và Learned Projection.
"""
import torch
import torch.nn as nn
from .resnet import ClientModel, ServerModel
from ..attacks.decoder import Decoder


class ClientWithBottleneck(nn.Module):
    """
    Client AR-TAPE / LightSplit:
    - Base: stem (Conv3->64, BN, ReLU) + layer1 (2 BasicBlocks, 64 channels) -> shape (B, 64, 32, 32)
    - Flatten: (B, 65536)
    - Bottleneck:
        * fixed_bottleneck=True: Phép chiếu trực giao cố định R in R^{D x k} (B7 mode F style, QR orthonormal)
        * fixed_bottleneck=False: Linear(D, k, bias=False) học được, khởi tạo trực chuẩn bằng QR.
    """
    def __init__(self, k=1024, D=65536, init_ortho=True, seed=42, fixed_bottleneck=False, in_channels=3, stem_stride=1):
        super().__init__()
        self.base_client = ClientModel(in_channels=in_channels, stem_stride=stem_stride)
        self.k = k
        self.D = D
        self.fixed_bottleneck = fixed_bottleneck

        g = torch.Generator().manual_seed(seed)
        A = torch.randn(D, k, generator=g)
        Q, _ = torch.linalg.qr(A, mode="reduced")  # Q in R^{D x k}

        if fixed_bottleneck:
            self.register_buffer("R", Q)
            self.bottleneck = None
        else:
            self.R = None
            self.bottleneck = nn.Linear(D, k, bias=False)
            if init_ortho:
                with torch.no_grad():
                    self.bottleneck.weight.copy_(Q.t())  # weight shape: [k, D]

    def get_bottleneck_weight(self):
        """Trả về ma trận trọng số chiếu dạng [k, D] để dùng cho Conv lift attacker."""
        if self.fixed_bottleneck:
            return self.R.t()
        return self.bottleneck.weight

    def forward(self, x):
        h = self.base_client(x)
        h_flat = h.view(h.size(0), -1)
        if self.fixed_bottleneck:
            zt = torch.matmul(h_flat, self.R)
        else:
            zt = self.bottleneck(h_flat)
        return zt

    def forward_features(self, x):
        h = self.base_client(x)
        h_flat = h.view(h.size(0), -1)
        if self.fixed_bottleneck:
            zt = torch.matmul(h_flat, self.R)
        else:
            zt = self.bottleneck(h_flat)
        return zt, h

    def load_state_dict(self, state_dict, strict=True):
        clean_sd = {}
        for k, v in state_dict.items():
            clean_sd[k] = v
        # Tương thích với B7 và các checkpoint không có tiền tố base_client.
        if not any(k.startswith("base_client.") for k in clean_sd.keys()):
            new_sd = {}
            for k, v in clean_sd.items():
                if k not in ["R", "bottleneck.weight"]:
                    new_sd[f"base_client.{k}"] = v
                else:
                    new_sd[k] = v
            clean_sd = new_sd
        # Tương thích chéo: nếu state_dict có 'R' nhưng model dùng 'bottleneck.weight'
        if "R" in clean_sd and hasattr(self, "bottleneck") and self.bottleneck is not None:
            self.bottleneck.weight.data.copy_(clean_sd["R"].t())
            clean_sd.pop("R", None)
        # Hoặc nếu state_dict có 'bottleneck.weight' nhưng model dùng 'R'
        elif "bottleneck.weight" in clean_sd and hasattr(self, "R") and self.R is not None:
            self.R.data.copy_(clean_sd["bottleneck.weight"].t())
            clean_sd.pop("bottleneck.weight", None)
        return super().load_state_dict(clean_sd, strict=False)


class ServerWithBottleneck(nn.Module):
    """
    Server AR-TAPE / LightSplit:
    - unprojector_mode:
        * 'linear_fixed': Toán tử nâng chiều trực giao R^T (0 tham số thừa, B7 mode F style).
                          z_hat = z_t @ R^T in R^{B x D} -> không làm hẹp rank, không trôi BatchNorm.
        * 'linear_learned': Tầng Linear(k, D, bias=False) học được, khởi tạo trực giao từ R^T.
        * 'mlp': Mạng MLP phi tuyến Linear(k, m) -> BN -> ReLU -> Linear(m, D) (B7 mode L style).
    - Reshape: (B, 64, 32, 32)
    - Base Server: ResNet-18 layer2..4 + AvgPool + FC
    """
    def __init__(self, k=1024, m=512, D=65536, num_classes=10, unprojector_mode="linear_fixed", ortho_matrix=None, seed=42):
        super().__init__()
        self.k = k
        self.m = m
        self.D = D
        self.unprojector_mode = unprojector_mode

        if unprojector_mode == "linear_fixed":
            if ortho_matrix is not None:
                self.register_buffer("R", ortho_matrix.clone())
            else:
                g = torch.Generator().manual_seed(seed)
                A = torch.randn(D, k, generator=g)
                Q, _ = torch.linalg.qr(A, mode="reduced")
                self.register_buffer("R", Q)
            self.unprojector = None
        elif unprojector_mode == "linear_learned":
            self.unprojector = nn.Linear(k, D, bias=False)
            if ortho_matrix is not None:
                Q = ortho_matrix
            else:
                g = torch.Generator().manual_seed(seed)
                A = torch.randn(D, k, generator=g)
                Q, _ = torch.linalg.qr(A, mode="reduced")
            with torch.no_grad():
                self.unprojector.weight.copy_(Q)  # [D, k] in PyTorch Linear(k, D)
        elif unprojector_mode == "mlp":
            self.unprojector = nn.Sequential(
                nn.Linear(k, m),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True),
                nn.Linear(m, D)
            )
        else:
            raise ValueError(f"Không hỗ trợ unprojector_mode='{unprojector_mode}'. Chọn 'linear_fixed', 'linear_learned', hoặc 'mlp'.")

        self.base_server = ServerModel(num_classes=num_classes) if hasattr(ServerModel, "__init__") and "num_classes" in ServerModel.__init__.__code__.co_varnames else ServerModel()

    def lift(self, zt):
        if zt.dim() > 2:
            zt = zt.view(zt.size(0), -1)
        if self.unprojector_mode == "linear_fixed":
            zh = torch.matmul(zt, self.R.t())
        else:
            zh = self.unprojector(zt)
        return zh

    def forward(self, zt):
        zh = self.lift(zt)
        zh_tensor = zh.view(zt.size(0), 64, 32, 32)
        return self.base_server(zh_tensor)

    def load_state_dict(self, state_dict, strict=True):
        clean_sd = {}
        for k, v in state_dict.items():
            clean_sd[k] = v
        if "R" in clean_sd and hasattr(self, "unprojector") and self.unprojector is not None and isinstance(self.unprojector, nn.Linear):
            self.unprojector.weight.data.copy_(clean_sd["R"])
            clean_sd.pop("R", None)
        elif "unprojector.weight" in clean_sd and hasattr(self, "R") and self.R is not None:
            self.R.data.copy_(clean_sd["unprojector.weight"])
            clean_sd.pop("unprojector.weight", None)
        return super().load_state_dict(clean_sd, strict=False)


class LearnedAdaptiveDecoder(nn.Module):
    """
    In-training Strong Adversary (~34M parameters):
    zt in R^{B x k} -> unprojector (Linear(k, 512) -> BN -> ReLU -> Linear(512, 65536) -> BN)
    -> reshape (B, 64, 32, 32) -> ConvDecoder(64, 3) -> x_rec in R^{B x 3 x 32 x 32}
    """
    def __init__(self, k=1024, D=65536, hidden_dim=512, out_channels=3):
        super().__init__()
        self.k = k
        self.D = D
        self.unprojector = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D),
            nn.BatchNorm1d(D)
        )
        self.conv_decoder = Decoder(in_channels=64, out_channels=out_channels)

    def forward(self, zt):
        b = zt.size(0)
        z_hat = self.unprojector(zt)
        z_hat_4d = z_hat.view(b, 64, 32, 32)
        x_rec = self.conv_decoder(z_hat_4d)
        return x_rec
