# Bước 3 — Module phòng thủ B2: DP-SGD (Differential Privacy SGD) cho Client trong Split Learning
import math
import torch
import torch.nn as nn


def compute_rdp_subsampled_gaussian(q, sigma, alpha):
    """
    Tính Rényi Differential Privacy (RDP) tại bậc alpha cho cơ chế Subsampled Gaussian.
    Tham khảo: Wang, Balle, Kasiviswanathan (2019) / Mironov et al. (2019).
    
    Tham số:
        q: Tỷ lệ lấy mẫu (batch_size / dataset_size)
        sigma: Hệ số nhiễu noise_multiplier (sigma_DP)
        alpha: Bậc Rényi (alpha > 1)
    """
    if sigma <= 0:
        return float('inf')
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2.0 * (sigma ** 2))

    # Xấp xỉ tight bound cho subsampled Gaussian mechanism với q << 1:
    c = math.exp(1.0 / (sigma ** 2)) - 1.0
    term2 = (alpha * (alpha - 1.0) / 2.0) * (q ** 2) * c
    if term2 < 0.1:
        return term2 / (alpha - 1.0)
    else:
        return math.log(1.0 + term2) / (alpha - 1.0)


def compute_dp_epsilon(epochs, batch_size=128, dataset_size=50000, noise_multiplier=1.0, delta=1e-5):
    """
    Tính ngân sách bảo mật (epsilon, delta)-DP tích lũy sau toàn bộ các epochs huấn luyện.
    Dùng phương pháp tối ưu hóa bậc alpha trên RDP.
    
    Tham số:
        epochs: Số epoch huấn luyện
        batch_size: Kích thước batch
        dataset_size: Tổng số mẫu tập huấn luyện (CIFAR-10 = 50,000)
        noise_multiplier: Hệ số nhiễu Gauss sigma_DP
        delta: Ngưỡng xác suất vi phạm riêng tư (mặc định 1e-5)
    
    Trả về:
        epsilon: Giá trị epsilon tối ưu (float)
    """
    if noise_multiplier <= 0:
        return float('inf')
    
    q = float(batch_size) / float(dataset_size)
    steps = int(epochs * math.ceil(dataset_size / batch_size))
    
    # Tập các bậc Rényi alpha ứng viên
    alphas = [1.1 + 0.1 * i for i in range(40)] + [5.0 + 1.0 * i for i in range(45)] + [50.0, 64.0, 100.0, 150.0]
    eps_candidates = []
    
    for alpha in alphas:
        if alpha <= 1.0:
            continue
        rdp_step = compute_rdp_subsampled_gaussian(q, noise_multiplier, alpha)
        rdp_total = steps * rdp_step
        # Đổi từ RDP sang (epsilon, delta)-DP:
        # eps(delta) = rdp_total + log(1/delta) / (alpha - 1)
        eps = rdp_total + math.log(1.0 / delta) / (alpha - 1.0)
        eps_candidates.append(eps)
        
    return min(eps_candidates) if eps_candidates else float('inf')


class DPSGDClientOptimizer:
    """
    Bộ tối ưu hóa DP-SGD cho Client trong Split Learning:
    1. Tiếp nhận gradient cut-layer dL/dz từ Server.
    2. Cắt ngưỡng gradient (Gradient Clipping) theo từng micro-batch / sample với ngưỡng C.
    3. Thêm nhiễu Gauss N(0, sigma_DP^2 * C^2 * I).
    4. Cập nhật trọng số Client qua optimizer bên dưới.
    """
    def __init__(self, client, optimizer, max_grad_norm=1.0, noise_multiplier=1.0, micro_batch_size=None):
        """
        Tham số:
            client: Mô hình ClientModel
            optimizer: Optimizer nền của Client (torch.optim.SGD, etc.)
            max_grad_norm: Ngưỡng cắt chuẩn gradient C (L2 norm)
            noise_multiplier: Cường độ nhiễu Gauss sigma_DP
            micro_batch_size: Kích thước micro-batch để cắt gradient.
                              Nếu None hoặc >= batch_size: cắt chuẩn theo toàn bộ batch.
        """
        self.client = client
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.micro_batch_size = micro_batch_size

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, x, grad_z):
        """
        Thực hiện 1 bước backward và cập nhật trọng số cho Client với bảo vệ DP-SGD.
        
        Tham số:
            x: Batch ảnh đầu vào của Client (B, 3, 32, 32)
            grad_z: Gradient dL/dz từ Server gửi về (B, 64, 32, 32)
            
        Trả về:
            stats: Dict thông tin chẩn đoán (grad_norm, noise_norm)
        """
        device = x.device
        batch_size = x.size(0)
        use_micro_batch = (
            self.micro_batch_size is not None 
            and 0 < self.micro_batch_size < batch_size
        )

        # Chế độ 1: Huấn luyện thông thường (không DP)
        if self.max_grad_norm <= 0.0 and self.noise_multiplier <= 0.0:
            self.optimizer.zero_grad()
            z = self.client(x)
            z.backward(grad_z)
            self.optimizer.step()
            return {"grad_norm": 0.0, "noise_norm": 0.0}

        # Chế độ 2: Cắt gradient theo toàn bộ batch (Batch-level clipping)
        if not use_micro_batch:
            self.optimizer.zero_grad()
            z = self.client(x)
            z.backward(grad_z)

            # Cắt chuẩn gradient cả mạng Client
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.client.parameters(), 
                max_norm=self.max_grad_norm
            )
            total_norm_val = float(total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm)

            # Thêm nhiễu Gauss chuẩn hóa theo batch size B
            noise_norm_val = 0.0
            if self.noise_multiplier > 0.0:
                noise_scale = (self.noise_multiplier * self.max_grad_norm) / float(batch_size)
                for p in self.client.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * noise_scale
                        p.grad.add_(noise)
                        noise_norm_val += noise.norm(2).item() ** 2
                noise_norm_val = math.sqrt(noise_norm_val)

            self.optimizer.step()
            return {"grad_norm": total_norm_val, "noise_norm": noise_norm_val}

        # Chế độ 3: Micro-batch / Per-sample gradient clipping
        # Tách batch thành các micro-batch, tính grad và clip từng phần
        m_size = self.micro_batch_size
        num_microbatches = (batch_size + m_size - 1) // m_size
        accum_grads = {p: torch.zeros_like(p.data) for p in self.client.parameters() if p.requires_grad}

        sum_norms = 0.0
        for i in range(0, batch_size, m_size):
            self.optimizer.zero_grad()
            x_mb = x[i : i + m_size]
            gz_mb = grad_z[i : i + m_size]

            z_mb = self.client(x_mb)
            z_mb.backward(gz_mb)

            # Tính L2 norm của micro-batch này
            mb_norm_sq = 0.0
            for p in self.client.parameters():
                if p.grad is not None:
                    mb_norm_sq += p.grad.norm(2).item() ** 2
            mb_norm = math.sqrt(mb_norm_sq)
            sum_norms += mb_norm

            # Hệ số cắt clip
            clip_coef = min(1.0, self.max_grad_norm / (mb_norm + 1e-6))

            # Tích lũy gradient đã cắt
            for p in self.client.parameters():
                if p.grad is not None:
                    accum_grads[p].add_(p.grad, alpha=clip_coef)

        # Gán gradient tích lũy + cộng nhiễu Gauss
        noise_norm_val = 0.0
        for p in self.client.parameters():
            if p.requires_grad:
                if self.noise_multiplier > 0.0:
                    noise = torch.randn_like(accum_grads[p]) * (self.noise_multiplier * self.max_grad_norm)
                    noise_norm_val += noise.norm(2).item() ** 2
                    p.grad = (accum_grads[p] + noise) / float(num_microbatches)
                else:
                    p.grad = accum_grads[p] / float(num_microbatches)

        noise_norm_val = math.sqrt(noise_norm_val) / float(num_microbatches)
        self.optimizer.step()

        return {"grad_norm": sum_norms / num_microbatches, "noise_norm": noise_norm_val}
