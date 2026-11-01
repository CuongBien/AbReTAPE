# -*- coding: utf-8 -*-
# Module đánh giá độ an toàn và chất lượng tái tạo (PSNR, SSIM, LPIPS)
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


def denormalize(x, mean, std):
    """
    Khôi phục ảnh chuẩn hóa về dải giá trị [0, 1].
    x: Tensor kích thước (B, C, H, W)
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=x.device, dtype=x.dtype)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def psnr_ssim(x, x_hat, mean, std):
    """
    Tính PSNR (dB) và SSIM trung bình cho batch ảnh x và x_hat.
    - x: Tensor ảnh gốc đã chuẩn hóa
    - x_hat: Tensor ảnh tái tạo từ Decoder
    Trả về: (mean_psnr, mean_ssim)
    """
    x_denorm = denormalize(x, mean, std).detach().cpu().numpy()
    x_hat_denorm = denormalize(x_hat, mean, std).detach().cpu().numpy()

    psnr_list, ssim_list = [], []
    for i in range(x_denorm.shape[0]):
        a = x_denorm[i].transpose(1, 2, 0)
        b = x_hat_denorm[i].transpose(1, 2, 0)
        psnr_val = peak_signal_noise_ratio(a, b, data_range=1.0)
        ssim_val = structural_similarity(a, b, channel_axis=2, data_range=1.0)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def get_lpips_fn(device="cpu", net="alex"):
    """
    Khởi tạo hàm tính LPIPS (Learned Perceptual Image Patch Similarity).
    """
    if not _HAS_LPIPS:
        return None
    try:
        loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
        loss_fn.eval()
        return loss_fn
    except Exception as e:
        print(f"[WARN] Không thể khởi tạo LPIPS: {e}")
        return None


def calculate_lpips(*args, **kwargs):
    """
    Tính khoảng cách cảm nhận LPIPS giữa x và x_hat.
    Hỗ trợ linh hoạt cả hai thứ tự tham số:
    1. calculate_lpips(x, x_hat, mean, std, lpips_fn)
    2. calculate_lpips(lpips_fn, x, x_hat, mean, std)
    Đầu vào cho lpips là ảnh dải [-1, 1].
    """
    if len(args) == 5:
        # Nếu arg đầu tiên không phải Tensor (là lpips_fn hoặc None)
        if not isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            lpips_fn, x, x_hat, mean, std = args
        else:
            x, x_hat, mean, std, lpips_fn = args
    elif len(args) >= 2:
        x, x_hat = args[0], args[1]
        mean = kwargs.get("mean", args[2] if len(args) > 2 else None)
        std = kwargs.get("std", args[3] if len(args) > 3 else None)
        lpips_fn = kwargs.get("lpips_fn", args[4] if len(args) > 4 else None)
    elif "lpips_fn" in kwargs:
        lpips_fn = kwargs.get("lpips_fn")
        x = kwargs.get("x")
        x_hat = kwargs.get("x_hat")
        mean = kwargs.get("mean")
        std = kwargs.get("std")
    else:
        return None

    if lpips_fn is None:
        return None

    try:
        x_01 = denormalize(x, mean, std)
        x_hat_01 = denormalize(x_hat, mean, std)

        x_norm = x_01 * 2.0 - 1.0
        x_hat_norm = x_hat_01 * 2.0 - 1.0

        with torch.no_grad():
            dist = lpips_fn(x_norm, x_hat_norm)
            return float(dist.mean().item())
    except Exception as e:
        print(f"[WARN LPIPS] Bỏ qua lỗi tính LPIPS: {e}")
        return None
