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


def recover_perm_from_adapter(A):
    """
    Khôi phục hoán vị kênh từ ma trận trọng số của Adapter 1x1 Conv:
    A shape [C_out, C_in] = [64, 64].
    A[r, c] là trọng số từ kênh vào c của z_perm tới kênh ra r của z_hat.
    Kênh vào c chứa z[pi[c]], do đó để đưa về kênh ra r = pi[c],
    hàng r = pi[c] sẽ có giá trị tuyệt đối lớn nhất:
        hat_pi[c] = argmax_r |A[r, c]|
    """
    if not torch.is_tensor(A):
        A = torch.tensor(A)
    return A.abs().argmax(dim=0).tolist()


@torch.no_grad()
def recover_perm_covariance(client, loader, perm, device, num_batches=20):
    """
    Thí nghiệm bổ trợ (Cách 2) — Channel Covariance Attack:
    Khôi phục hoán vị pi từ profile phương sai của 64 kênh trong z và z'.
    Sigma_z' = P_pi * Sigma_z * P_pi^T.
    Không cần huấn luyện (0 epochs), khôi phục tức thì qua thống kê phân phối!
    """
    client.eval()
    all_z = []
    batch_count = 0

    for x, _ in loader:
        x = x.to(device)
        z = client(x)  # [B, 64, H, W]
        # Gom spatial dimensions lại để tính phương sai từng kênh
        z_flat = z.permute(1, 0, 2, 3).reshape(64, -1)
        all_z.append(z_flat)
        batch_count += 1
        if batch_count >= num_batches:
            break

    full_z = torch.cat(all_z, dim=1)  # [64, Total_Pixels]
    var_clean = full_z.var(dim=1).cpu().numpy()  # [64]

    # z' có kênh c là kênh perm[c] của z
    if torch.is_tensor(perm):
        perm_list = perm.tolist()
    else:
        perm_list = list(perm)
    var_perm = var_clean[perm_list]

    # Với mỗi kênh c của z', tìm kênh c' của clean z có phương sai gần nhất
    cost = np.abs(var_perm[:, None] - var_clean[None, :])  # [64, 64]
    perm_hat = np.argmin(cost, axis=1).tolist()
    acc = match_accuracy(perm_list, perm_hat)
    return perm_hat, acc
