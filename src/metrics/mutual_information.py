# Module ước lượng Thông tin Tương hỗ (Mutual Information - MI)
# Phục vụ phân tích Information Leakage: I(X; Z') và Nuisance Information I(X; Z') - I(Y; Z')
import torch
import torch.nn as nn
import torch.nn.functional as F
from .distance_correlation import distance_correlation


class MutualInformationEstimator(nn.Module):
    """
    Ước lượng Thông tin Tương hỗ I(X; Z') dựa trên nguyên lý Donsker-Varadhan / MINE (Belghazi et al. 2018)
    hoặc InfoNCE lower-bound:
    I(X; Z') >= E_p [T(x, z)] - log E_q [exp(T(x, z_rand))]
    """
    def __init__(self, x_dim=None, z_dim=None, hidden_dim=256, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, z):
        # Thu gọn chiều không gian để ước lượng ổn định
        b = x.size(0)
        if x.dim() == 4:
            x_down = F.adaptive_avg_pool2d(x, (4, 4)).view(b, -1)
        else:
            x_down = x.view(b, -1)

        if z.dim() == 4:
            z_down = F.adaptive_avg_pool2d(z, (4, 4)).view(b, -1)
        else:
            z_down = z.view(b, -1)

        # Cặp mẫu khớp (joint distribution P_XZ)
        joint = torch.cat([x_down, z_down], dim=1)
        t_joint = self.net(joint)

        # Cặp mẫu ngẫu nhiên (marginal distribution P_X x P_Z)
        z_shuffled = z_down[torch.randperm(b)]
        marginal = torch.cat([x_down, z_shuffled], dim=1)
        t_marginal = self.net(marginal)

        # DV bound
        mi_lb = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(torch.clamp(t_marginal, max=20.0))) + 1e-8)
        return mi_lb


def estimate_dcor_and_mi(x, z, mi_estimator=None):
    """
    Tính đồng thời Distance Correlation dCor(X, Z') và ước lượng Mutual Information.
    """
    dcor_val = distance_correlation(x, z).item()
    mi_val = None
    if mi_estimator is not None:
        with torch.no_grad():
            mi_val = mi_estimator(x, z).item()

    return {
        "distance_correlation": float(dcor_val),
        "mutual_information_bound": float(mi_val) if mi_val is not None else None
    }
