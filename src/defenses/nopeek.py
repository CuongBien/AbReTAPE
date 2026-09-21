# Baseline B3: NoPeek — Giảm thiểu thông tin rò rỉ qua phạt Distance Correlation (dCor)
# Tham khảo: Vepakomma et al., "NoPeek: Information leakage reduction to share activations in distributed deep learning", 2020.
import torch
import torch.nn as nn
from ..metrics.distance_correlation import distance_correlation


class NoPeekDefense(nn.Module):
    """
    Module quản lý phạt khoảng cách tương quan NoPeek:
    L_Client = L_task(logits, Y) + alpha * dCor(X, Z')
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, z):
        # NoPeek không làm méo biểu diễn z trong forward pass
        return z

    def compute_loss(self, x, z):
        """
        Tính lượng phạt dCor(X, Z') nhân với siêu tham số alpha:
        """
        if self.alpha <= 0.0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        dcor = distance_correlation(x, z)
        return self.alpha * dcor

    def extra_repr(self):
        return f"alpha={self.alpha}"
