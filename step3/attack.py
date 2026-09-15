# Bước 3 — Huấn luyện và đánh giá Decoder tấn công tái tạo thích ứng trên IR có phòng thủ
import sys
import os
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP1_DIR = os.path.join(PROJECT_ROOT, "step1")

for path in reversed([SCRIPT_DIR, STEP0_DIR if 'STEP0_DIR' in locals() else os.path.join(PROJECT_ROOT, "step0"), STEP1_DIR, PROJECT_ROOT]):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from metrics import psnr_ssim, calculate_lpips


def train_adaptive_decoder_epoch(client, decoder, defense, loader, opt, criterion, device):
    """
    Huấn luyện Decoder tấn công trong 1 epoch.
    Client và cơ chế phòng thủ bị đóng băng: kẻ tấn công chỉ nhận được z' = defense(client(x)).
    min_phi E[ || D_phi(defense(F_c(x))) - x ||^2 ]
    """
    client.eval()
    defense.eval()
    decoder.train()
    total_loss = 0.0
    total_samples = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        batch_size = x.size(0)

        opt.zero_grad()

        with torch.no_grad():
            z = client(x)
            z_def = defense(z)

        x_hat = decoder(z_def)
        loss = criterion(x_hat, x)

        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_attack(client, decoder, defense, loader, device, mean, std, criterion=None, lpips_fn=None):
    """
    Đánh giá độ an toàn (Security) của cơ chế phòng thủ:
    Đo lường khả năng tái tạo ảnh x_hat từ IR nhiễu z' trên Test set.
    Trả về: (mean_mse, mean_psnr, mean_ssim, mean_lpips)
    """
    client.eval()
    defense.eval()
    decoder.eval()

    total_mse = 0.0
    total_samples = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    has_lpips = lpips_fn is not None

    if criterion is None:
        criterion = nn.MSELoss()

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        batch_size = x.size(0)

        z = client(x)
        z_def = defense(z)
        x_hat = decoder(z_def)

        loss = criterion(x_hat, x)
        total_mse += loss.item() * batch_size

        batch_psnr, batch_ssim = psnr_ssim(x, x_hat, mean, std)
        psnr_sum += batch_psnr * batch_size
        ssim_sum += batch_ssim * batch_size

        if has_lpips:
            batch_lpips = calculate_lpips(x, x_hat, mean, std, lpips_fn)
            if batch_lpips is not None:
                lpips_sum += batch_lpips * batch_size

        total_samples += batch_size

    mean_mse = total_mse / total_samples
    mean_psnr = psnr_sum / total_samples
    mean_ssim = ssim_sum / total_samples
    mean_lpips = (lpips_sum / total_samples) if has_lpips else None

    return mean_mse, mean_psnr, mean_ssim, mean_lpips


# Alias tương thích ngược
train_decoder_epoch = train_adaptive_decoder_epoch
