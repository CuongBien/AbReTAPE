# Module tính toán Khoảng cách Tương quan (Distance Correlation - dCor) trên PyTorch
# Sử dụng phương pháp U-centering (Székely & Rizzo 2014) không thiên lệch (unbiased),
# giải quyết triệt để hiện tượng suy biến ma trận đường chéo trong không gian nhiều chiều.
import torch


def u_centering(dist_matrix):
    """
    Thực hiện định tâm U-centering (Székely & Rizzo, 2014) cho ma trận khoảng cách:
    Loại bỏ bias đường chéo D_ii = 0 trong không gian nhiều chiều.
    """
    n = dist_matrix.size(0)
    if n <= 3:
        # Nếu batch quá nhỏ, dùng double centering thông thường
        row_mean = dist_matrix.mean(dim=1, keepdim=True)
        col_mean = dist_matrix.mean(dim=0, keepdim=True)
        grand_mean = dist_matrix.mean()
        return dist_matrix - row_mean - col_mean + grand_mean

    # Đảm bảo đường chéo bằng 0
    dist = dist_matrix.clone()
    dist.fill_diagonal_(0.0)

    row_sum = dist.sum(dim=1, keepdim=True)  # (n, 1)
    col_sum = dist.sum(dim=0, keepdim=True)  # (1, n)
    total_sum = dist.sum()                   # scalar

    # Công thức U-centering
    u_mat = dist - (row_sum / (n - 2.0)) - (col_sum / (n - 2.0)) + (total_sum / ((n - 1.0) * (n - 2.0)))
    u_mat.fill_diagonal_(0.0)
    return u_mat


def distance_covariance_sq(x, z):
    """
    Tính ước lượng không thiên lệch (U-statistic) của bình phương hiệp phương sai khoảng cách dCov^2(X, Z).
    """
    b = x.size(0)
    x_flat = x.view(b, -1)
    z_flat = z.view(b, -1)

    # Tính ma trận khoảng cách đôi một Euclidean
    dist_x = torch.cdist(x_flat, x_flat, p=2)
    dist_z = torch.cdist(z_flat, z_flat, p=2)

    # Định tâm U-centering
    a_u = u_centering(dist_x)
    b_u = u_centering(dist_z)

    # Chuẩn hóa mẫu n*(n-3)
    denom = float(b * (b - 3.0)) if b > 3 else float(b * b)
    dcov2 = (a_u * b_u).sum() / denom
    return dcov2, a_u, b_u


def distance_correlation(x, z, eps=1e-12):
    """
    Tính khoảng cách tương quan không thiên lệch dCor(X, Z) in [0, 1]:
    dCor(X, Z) = sqrt( clamp(dCov^2(X, Z), min=0) / sqrt(dVar^2(X) * dVar^2(Z)) )

    - x: Tensor (B, ...) - ảnh đầu vào Client
    - z: Tensor (B, ...) - biểu diễn trung gian smashed data tại cut layer
    - eps: hằng số chống chia cho 0 để bảo toàn gradient
    """
    b = x.size(0)
    if b <= 3:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    dcov2, a_u, b_u = distance_covariance_sq(x, z)
    denom = float(b * (b - 3.0))

    dvar_x2 = (a_u * a_u).sum() / denom
    dvar_z2 = (b_u * b_u).sum() / denom

    dvar_prod = torch.clamp(dvar_x2 * dvar_z2, min=eps)
    dcor_sq = torch.clamp(dcov2, min=0.0) / (torch.sqrt(dvar_prod) + 1e-8)
    dcor = torch.sqrt(torch.clamp(dcor_sq, min=0.0, max=1.0) + 1e-10)
    return dcor
