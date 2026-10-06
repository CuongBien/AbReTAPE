# Module Đề xuất: AR-TAPE (Absorption-Resistant Task-Aware Perceptual Encoding)
# Hiện thực hóa Novelty N1, N2 và RQ1: Chiếu không gian con phi khả nghịch P_task = V_task * V_task^T
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubspaceProjector(nn.Module):
    """
    Toán tử chiếu trực giao lên không gian con tác vụ P_task = V_task * V_task^T.
    V_task có kích thước [C, k] với k < C (không gian con thu gọn, rank-deficient).
    Thành phần trực giao P_perp * z = (I - P_task) * z chứa cấu trúc hình học tri giác
    sẽ bị triệt tiêu hoàn toàn, khiến ánh xạ ngược E^-1 trở nên bất khả (phi khả nghịch).
    """
    def __init__(self, channels=64, subspace_dim=32):
        super().__init__()
        self.channels = channels
        self.subspace_dim = min(subspace_dim, channels - 1)  # Đảm bảo k < C

        # Ma trận cơ sở trực chuẩn V_task
        init_mat = torch.randn(channels, self.subspace_dim)
        q, _ = torch.linalg.qr(init_mat)
        self.v_basis = nn.Parameter(q)

    def get_projection_matrix(self):
        # Đảm bảo trực giao hóa bằng QR hoặc Gram-Schmidt khi tính P_task
        q, _ = torch.linalg.qr(self.v_basis)
        p_task = torch.matmul(q, q.t())  # [C, C]
        return p_task

    def forward(self, z):
        # z: [B, C, H, W]
        b, c, h, w = z.shape
        p_task = self.get_projection_matrix()

        z_permuted = z.permute(0, 2, 3, 1).reshape(-1, c)  # [B*H*W, C]
        z_proj = torch.matmul(z_permuted, p_task)           # [B*H*W, C]
        z_proj = z_proj.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return z_proj


class AR_TAPE(nn.Module):
    """
    Kiến trúc AR-TAPE tổng quát:
    1. Subspace Projection (P_task z) loại bỏ vĩnh viễn không gian P_perp z.
    2. Phi tuyến hóa nén kênh (Non-invertible channel squashing / quantization).
    3. Huấn luyện nhận biết nhiệm vụ (Task-Aware): Giữ nguyên gradient cho tác vụ Y
       trong khi triệt tiêu tối đa rò rỉ cho Feature Inversion & FSHA.
    """
    def __init__(self, in_channels=64, subspace_dim=32, non_linear_dropout=0.1):
        super().__init__()
        self.projector = SubspaceProjector(channels=in_channels, subspace_dim=subspace_dim)
        self.norm = nn.BatchNorm2d(in_channels, affine=True)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(p=non_linear_dropout) if non_linear_dropout > 0 else nn.Identity()

    def forward(self, z):
        # 1. Chiếu lên không gian con tác vụ (triệt tiêu rank)
        z_task = self.projector(z)
        # 2. Chuẩn hóa & phi tuyến tính phá vỡ quan hệ affine tuyến tính của Adapter Conv 1x1
        z_encoded = self.activation(self.norm(z_task))
        z_encoded = self.dropout(z_encoded)
        return z_encoded

    def get_orthogonality_loss(self):
        """
        Loss phạt nếu các cột của V_task lệch khỏi trực chuẩn: || V^T V - I_k ||_F^2
        """
        q, _ = torch.linalg.qr(self.projector.v_basis)
        gram = torch.matmul(self.projector.v_basis.t(), self.projector.v_basis)
        eye = torch.eye(self.projector.subspace_dim, device=self.projector.v_basis.device)
        return F.mse_loss(gram, eye)
