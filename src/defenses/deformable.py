# Baseline B5: Deformable Operators (Task-aware bằng thiết kế, Kiya et al. 2024)
# Thực hiện biến dạng hình học không gian (Spatial Grid Warping / Deformable transformation)
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableOperatorDefense(nn.Module):
    """
    Baseline B5 (Kiya et al. 2024):
    Áp dụng phép biến dạng lưới tọa độ không gian (spatial grid warping) ngẫu nhiên có tham số hóa,
    làm vặn xoắn hình học của feature map z, phá vỡ sự tương ứng pixel-to-pixel giữa input và smashed data.
    """
    def __init__(self, channels=64, distortion_scale=0.2, seed=42):
        super().__init__()
        self.channels = channels
        self.distortion_scale = distortion_scale
        self.seed = seed
        self.grid_offset = None

    def _init_offset(self, h, w, device):
        if self.grid_offset is None:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            # Tạo offset ngẫu nhiên trơn tru (smooth random displacement)
            raw_noise = torch.randn(1, 2, h // 4, w // 4, generator=g, device=device)
            smooth_offset = F.interpolate(raw_noise, size=(h, w), mode="bicubic", align_corners=True)
            self.grid_offset = smooth_offset * self.distortion_scale

    def forward(self, z):
        b, c, h, w = z.shape
        self._init_offset(h, w, z.device)

        # Tạo lưới cơ sở chuẩn [-1, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, h, device=z.device),
            torch.linspace(-1, 1, w, device=z.device),
            indexing="ij"
        )
        base_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)  # [B, H, W, 2]

        offset = self.grid_offset.permute(0, 2, 3, 1).repeat(b, 1, 1, 1)  # [B, H, W, 2]
        deformed_grid = torch.clamp(base_grid + offset, -1.0, 1.0)

        z_deformed = F.grid_sample(z, deformed_grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        return z_deformed
