# Thuật toán khôi phục hoán vị kênh từ trọng số Server, Adapter hoặc Profile Covariance
import torch
import numpy as np


def compute_cost_matrix(W, W_ref):
    """
    Tính ma trận khoảng cách Euclidean bình phương giữa các kênh đầu vào của W và W_ref:
    cost_matrix[c, cp] = || W[:, c] - W_ref[:, cp] ||^2
    """
    if torch.is_tensor(W):
        W = W.detach().cpu()
    if torch.is_tensor(W_ref):
        W_ref = W_ref.detach().cpu()

    C_in = W.shape[1]
    w_flat = W.permute(1, 0, 2, 3).reshape(C_in, -1)
    w_ref_flat = W_ref.permute(1, 0, 2, 3).reshape(C_in, -1)

    w_norm_sq = (w_flat ** 2).sum(dim=1, keepdim=True)
    w_ref_norm_sq = (w_ref_flat ** 2).sum(dim=1, keepdim=True).t()
    cross = torch.mm(w_flat, w_ref_flat.t())

    dist_matrix = (w_norm_sq + w_ref_norm_sq - 2.0 * cross).clamp(min=0.0)
    return dist_matrix.numpy()


def compute_cosine_similarity_matrix(W, W_ref):
    """
    Tính ma trận tương đồng Cosine giữa các kênh đầu vào của W và W_ref.
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
    Khôi phục hoán vị kênh bằng cách tìm kênh có trọng số gần nhất:
    hat_pi(c) = argmin_cp || W[:, c] - W_ref[:, cp] ||^2
    """
    cost_matrix = compute_cost_matrix(W, W_ref)
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
    Khôi phục hoán vị kênh từ ma trận trọng số Adapter: hat_pi[c] = argmax_r |A[r, c]|
    """
    if not torch.is_tensor(A):
        A = torch.tensor(A)
    return A.abs().argmax(dim=0).tolist()


@torch.no_grad()
def recover_perm_covariance(client, loader, perm, device, num_batches=20):
    """
    Channel Covariance Statistical Attack:
    Khôi phục hoán vị pi từ profile phương sai của 64 kênh trong z và z'.
    """
    client.eval()
    all_z = []
    batch_count = 0

    for x, _ in loader:
        x = x.to(device)
        z = client(x)
        z_flat = z.permute(1, 0, 2, 3).reshape(64, -1)
        all_z.append(z_flat)
        batch_count += 1
        if batch_count >= num_batches:
            break

    full_z = torch.cat(all_z, dim=1)
    var_clean = full_z.var(dim=1).cpu().numpy()

    if torch.is_tensor(perm):
        perm_list = perm.tolist()
    else:
        perm_list = list(perm)
    var_perm = var_clean[perm_list]

    cost = np.abs(var_perm[:, None] - var_clean[None, :])
    perm_hat = np.argmin(cost, axis=1).tolist()
    acc = match_accuracy(perm_list, perm_hat)
    return perm_hat, acc
