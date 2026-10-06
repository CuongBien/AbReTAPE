# Module Baseline B7: Fixed Orthogonal Projection (LightSplit-style, arXiv:2605.13265)
# Đối chiếu full-text (Cihangiroglu et al., 05/2026):
# - R: (D, k) cột trực chuẩn, QR mỏng của Gaussian (R^T R = I_k)
# - LightSplit-F: z_hat = R z_t (server không tham số) -> reshape (64, 32, 32)
# - LightSplit-L: MLP k -> m -> D có BN -> reshape (64, 32, 32)
# - Testbed vanilla SL giữ lambda_WCC = 0 vì nhãn ở server (U-shaped topology mới có nhãn ở client)
import torch
import torch.nn as nn


class FixedOrthoProjection(nn.Module):
    """
    Toán tử chiếu trực giao cố định ngẫu nhiên R: R in R^{D x k}, R^T R = I_k.
    Tại Client: z in R^{B x D} -> z_t = z @ R in R^{B x k}.
    Tại Server (F-mode): z_hat = z_t @ R^T in R^{B x D}.
    """
    def __init__(self, D=65536, k=1024, seed=42, device=None):
        super().__init__()
        self.D = D
        self.k = k
        self.seed = seed

        # Dùng generator trên CPU để đảm bảo ma trận A tái lập 100% nhất quán giữa CPU và GPU
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(D, k, generator=g)

        # Tính toán QR trên GPU nếu có để tăng tốc (nhanh hơn CPU 50x)
        use_cuda = (device is not None and torch.device(device).type == "cuda") or (device is None and torch.cuda.is_available())
        qr_device = torch.device("cuda" if use_cuda else "cpu")

        if qr_device.type == "cuda":
            A = A.to(qr_device)
            Q, _ = torch.linalg.qr(A, mode="reduced")
        else:
            Q, _ = torch.linalg.qr(A, mode="reduced")

        # Đăng ký R dưới dạng buffer để tự động di chuyển theo model.to(device)
        self.register_buffer("R", Q)

    def forward(self, z_flat):
        """
        Client projection: z_flat (B, D) -> z_t = z_flat @ R (B, k).
        Nếu đầu vào là (B, C, H, W), tự động flatten thành (B, D).
        """
        if z_flat.dim() > 2:
            z_flat = z_flat.view(z_flat.size(0), -1)
        return torch.matmul(z_flat, self.R)

    def lift(self, zt):
        """
        Server lift (Mode F): z_t (B, k) -> z_hat = z_t @ R^T (B, D).
        """
        return torch.matmul(zt, self.R.t())

    def get_orthogonality_error(self):
        """
        Kiểm tra độ lệch trực chuẩn || R^T R - I_k ||_max
        """
        gram = torch.matmul(self.R.t(), self.R)
        eye = torch.eye(self.k, device=self.R.device)
        return (gram - eye).abs().max().item()

    def get_memory_bytes(self):
        return self.R.numel() * self.R.element_size()


class ProjectionMLP(nn.Module):
    """
    LightSplit-L: MLP k -> m -> D có BatchNorm1d:
    Linear(k, m) -> BatchNorm1d(m) -> ReLU() -> Linear(m, D).
    """
    def __init__(self, k, m, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, m),
            nn.BatchNorm1d(m),
            nn.ReLU(inplace=True),
            nn.Linear(m, D)
        )

    def forward(self, zt):
        return self.net(zt)


class SplitProjection(nn.Module):
    """
    Bọc projection + Base Server Model cho Split Learning:
    - mode 'F': z_hat = R z_t (0 tham số bổ sung) -> reshape (64, 32, 32) -> server gốc.
    - mode 'L': MLP k -> m -> D (học được) -> reshape (64, 32, 32) -> server gốc.
    """
    def __init__(self, proj: FixedOrthoProjection, base_server: nn.Module, mode="F", m=512, shape=(64, 32, 32)):
        super().__init__()
        self.proj = proj
        self.base_server = base_server
        self.mode = mode.upper()
        self.shape = shape
        self.encoder = ProjectionMLP(proj.k, m, proj.D) if self.mode == "L" else None

    def forward(self, zt):
        if self.mode == "L":
            zh = self.encoder(zt)
        else:
            zh = self.proj.lift(zt)
        # Reshape z_hat về shape gốc (B, 64, 32, 32) rồi chuyển tiếp vào server
        zh_tensor = zh.view(zt.size(0), *self.shape)
        return self.base_server(zh_tensor)


def wcc_loss(zt, y):
    """
    Within-Class Compaction loss (WCC) theo Equation (Section IV-D) của LightSplit.
    CHỈ dùng khi nhãn y ở client (topology U-shaped).
    Trong vanilla testbed (nhãn ở server), lambda_WCC = 0.
    """
    total = 0.0
    unique_classes = y.unique()
    for c in unique_classes:
        zc = zt[y == c]
        if zc.size(0) > 1:
            total = total + ((zc - zc.mean(0)) ** 2).sum(1).mean()
    return total
