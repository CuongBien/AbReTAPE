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


def calculate_lpips(x, x_hat, mean, std, lpips_fn):
    """
    Tính khoảng cách cảm nhận LPIPS giữa x và x_hat.
    Đầu vào cho lpips là ảnh dải [-1, 1].
    """
    if lpips_fn is None:
        return None

    x_01 = denormalize(x, mean, std)
    x_hat_01 = denormalize(x_hat, mean, std)

    x_norm = x_01 * 2.0 - 1.0
    x_hat_norm = x_hat_01 * 2.0 - 1.0

    with torch.no_grad():
        dist = lpips_fn(x_norm, x_hat_norm)
        return float(dist.mean().item())
