# Bước 2 — Hoán vị kênh cố định tại cut layer (Mã hóa khả nghịch)
import torch
import torch.nn as nn


def random_perm(num_channels, seed=42):
    """
    Sinh một hoán vị ngẫu nhiên cố định cho num_channels kênh.
    Sử dụng torch.Generator với seed cụ thể để đảm bảo tính tái lập.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(num_channels, generator=g)


class ChannelPermute(nn.Module):
    """
    Lớp hoán vị kênh z' = E(z), trong đó:
    z'[:, c, :, :] = z[:, perm[c], :, :]
    Được chèn vào ngay tại cut layer giữa Client và Server.
    """
    def __init__(self, perm):
        super().__init__()
        if not isinstance(perm, torch.Tensor):
            perm = torch.tensor(perm, dtype=torch.long)
        self.register_buffer("perm", perm)

    def forward(self, z):
        return z[:, self.perm, :, :]

    def inverse(self, z_prime):
        """
        Phép giải mã ngược lý thuyết z = E^(-1)(z')
        """
        inv_perm = torch.empty_like(self.perm)
        inv_perm[self.perm] = torch.arange(self.perm.size(0), device=self.perm.device)
        return z_prime[:, inv_perm, :, :]
