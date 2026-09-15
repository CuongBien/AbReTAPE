# Bước 2 — Khôi phục hoán vị kênh từ trọng số lớp đầu của Server
import torch
import numpy as np


def compute_cost_matrix(W, W_ref):
    """
    Tính ma trận khoảng cách Euclidean bình phương giữa các kênh đầu vào của W và W_ref.
    W:      Trọng số Conv lớp đầu server đã hấp thụ hoán vị, shape [C_out, C_in, k, k]
    W_ref:  Trọng số Conv lớp đầu server tham chiếu (huấn luyện sạch ở Bước 0)
    Trả về: cost_matrix shape [C_in, C_in], trong đó:
            cost_matrix[c, cp] = || W[:, c] - W_ref[:, cp] ||^2
    """
    if torch.is_tensor(W):
        W = W.detach().cpu()
    if torch.is_tensor(W_ref):
        W_ref = W_ref.detach().cpu()

    C_in = W.shape[1]
    # Làm phẳng thành [C_in, C_out * k * k]
    w_flat = W.permute(1, 0, 2, 3).reshape(C_in, -1)
    w_ref_flat = W_ref.permute(1, 0, 2, 3).reshape(C_in, -1)

    # Tính khoảng cách Euclidean bình phương theo batch
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
    w_norm_sq = (w_flat ** 2).sum(dim=1, keepdim=True)        # [C_in, 1]
    w_ref_norm_sq = (w_ref_flat ** 2).sum(dim=1, keepdim=True).t()  # [1, C_in]
    cross = torch.mm(w_flat, w_ref_flat.t())                  # [C_in, C_in]

    dist_matrix = (w_norm_sq + w_ref_norm_sq - 2.0 * cross).clamp(min=0.0)
    return dist_matrix.numpy()


def compute_cosine_similarity_matrix(W, W_ref):
    """
    Tính ma trận tương đồng Cosine giữa các kênh đầu vào của W và W_ref.
    Trả về: sim_matrix shape [C_in, C_in] trong khoảng [-1, 1]
    """
    if torch.is_tensor(W):
        W = W.detach().cpu()
    if torch.is_tensor(W_ref):
        W_ref = W_ref.detach().cpu()

    C_in = W.shape[1]
    w_flat = W.permute(1, 0, 2, 3).reshape(C_in, -1)
    w_ref_flat = W_ref.permute(1, 0, 2, 3).reshape(C_in, -1)

    w_norm = torch.nn.functional.normalize(w_flat, p=2, dim=1)
    w_ref_norm = torch.nn.functional.normalize(w_ref_flat, p=2, dim=1)

    sim_matrix = torch.mm(w_norm, w_ref_norm.t())
    return sim_matrix.numpy()


def recover_perm(W, W_ref):
    """
    Khôi phục hoán vị kênh: với mỗi kênh đầu vào c của Server,
    tìm kênh c' của Server tham chiếu có trọng số gần nhất:
    hat_pi(c) = argmin_cp || W[:, c] - W_ref[:, cp] ||^2
    """
    cost_matrix = compute_cost_matrix(W, W_ref)
    # Với mỗi hàng c, lấy cột cp có khoảng cách nhỏ nhất
    perm_hat = np.argmin(cost_matrix, axis=1).tolist()
    return perm_hat


def match_accuracy(perm_true, perm_hat):
    """
    Tính tỷ lệ phần trăm khớp chính xác giữa hoán vị thật và hoán vị khôi phục.
    """
    if torch.is_tensor(perm_true):
        perm_true = perm_true.tolist()
    if torch.is_tensor(perm_hat):
        perm_hat = perm_hat.tolist()

    correct = sum(1 for a, b in zip(perm_true, perm_hat) if a == b)
    return float(correct) / len(perm_true)
