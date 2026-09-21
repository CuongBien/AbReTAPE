# Vòng lặp huấn luyện và đánh giá bộ giải mã Feature Inversion Attack (hỗ trợ cả sạch và có defense)
import torch
import torch.nn as nn
from ..metrics.reconstruction import psnr_ssim, calculate_lpips


def train_inversion_epoch(client, decoder, loader, opt, criterion, device, defense=None):
    """
    Huấn luyện Decoder tái tạo qua 1 epoch trong khi Client F_c bị đóng băng.
    Nếu defense is not None: z' = defense(client(x)), Decoder học đảo ngược z'.
    min_phi E[ || D_phi(defense(F_c(x))) - x ||^2 ]
    """
    client.eval()
    if defense is not None:
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
            if defense is not None:
                z = defense(z)

        x_hat = decoder(z)
        loss = criterion(x_hat, x)

        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_inversion(client, decoder, loader, device, mean, std, defense=None, criterion=None, lpips_fn=None):
    """
    Đánh giá độ an toàn (Security) đối phó với kẻ tấn công tái tạo thụ động:
    Đo lường khả năng tái tạo ảnh x_hat từ biểu diễn trung gian trên Test set.
    Trả về: (mean_mse, mean_psnr, mean_ssim, mean_lpips)
    """
    client.eval()
    if defense is not None:
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
        if defense is not None:
            z = defense(z)
        x_hat = decoder(z)

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


# Aliases tương thích ngược với code cũ
train_decoder_epoch = train_inversion_epoch
train_adaptive_decoder_epoch = train_inversion_epoch
evaluate_attack = evaluate_inversion
