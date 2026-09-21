#!/usr/bin/env python3
# Kiểm thử đơn vị tự động toàn diện (Comprehensive Smoke Tests) cho toàn bộ hệ thống AbReTAPE
import os
import sys
import torch
import torch.nn as nn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel, ServerModel, resnet18_cifar
from src.defenses import random_perm, ChannelPermute, Adapter, GaussianNoise, DPSGDClientOptimizer, compute_dp_epsilon
from src.attacks import Decoder, recover_perm, recover_perm_from_adapter, match_accuracy
from src.metrics import psnr_ssim, denormalize, get_lpips_fn, calculate_lpips
from src.training import train_sl_epoch, evaluate_sl
from src.utils import plot_training_curves


def test_models_and_sl(device):
    print("\n[TEST 1/5] KIỂM THỬ MÔ HÌNH VÀ SPLIT LEARNING FORWARD/BACKWARD...")
    client = ClientModel().to(device)
    server = ServerModel().to(device)

    # 1. Kiểm tra shapes
    x_dummy = torch.randn(2, 3, 32, 32, device=device)
    z = client(x_dummy)
    assert z.shape == (2, 64, 32, 32), f"Lỗi shape z: {z.shape}"
    logits = server(z)
    assert logits.shape == (2, 10), f"Lỗi shape logits: {logits.shape}"
    print("  -> Kiểm tra kích thước z [2, 64, 32, 32] và logits [2, 10]: ĐẠT!")

    # 2. Kiểm tra backward qua biên giới
    opt_c = torch.optim.SGD(client.parameters(), lr=0.01)
    opt_s = torch.optim.SGD(server.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    y_dummy = torch.tensor([0, 1], device=device)

    dummy_loader = [(x_dummy, y_dummy)]
    train_loss, train_acc = train_sl_epoch(client, server, dummy_loader, opt_c, opt_s, criterion, device)
    val_loss, val_acc = evaluate_sl(client, server, dummy_loader, device, criterion=criterion)
    print("  -> Kiểm tra Split Learning forward & backward boundary: ĐẠT!")


def test_attacks_and_metrics(device):
    print("\n[TEST 2/5] KIỂM THỬ DECODER TẤN CÔNG VÀ BỘ ĐỘ ĐO (PSNR, SSIM, LPIPS)...")
    decoder = Decoder(in_channels=64, out_channels=3).to(device)
    z_dummy = torch.randn(2, 64, 32, 32, device=device)
    x_rec = decoder(z_dummy)
    assert x_rec.shape == (2, 3, 32, 32), f"Lỗi shape x_rec: {x_rec.shape}"

    # Kiểm tra PSNR, SSIM
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    x_clean = torch.rand(2, 3, 32, 32, device=device)
    psnr_val, ssim_val = psnr_ssim(x_clean, x_clean, mean, std)
    assert psnr_val > 50.0 and ssim_val > 0.99, f"PSNR/SSIM ảnh giống hệt không chính xác: {psnr_val}, {ssim_val}"
    print(f"  -> Kiểm tra tính toán PSNR/SSIM ({psnr_val:.1f} dB, {ssim_val:.4f}): ĐẠT!")

    lpips_fn = get_lpips_fn(device=device)
    if lpips_fn is not None:
        lpips_dist = calculate_lpips(x_clean, x_clean, mean, std, lpips_fn)
        assert lpips_dist < 1e-4, f"LPIPS ảnh giống hệt phải xấp xỉ 0: {lpips_dist}"
        print(f"  -> Kiểm tra tính toán LPIPS ({lpips_dist:.6f}): ĐẠT!")
    else:
        print("  -> Bỏ qua LPIPS (chưa cài đặt hoặc chạy CPU).")


def test_defenses_step2(device):
    print("\n[TEST 3/5] KIỂM THỬ HOÁN VỊ KÊNH, ADAPTER VÀ KHÔI PHỤC HẤP THỤ...")
    perm = random_perm(64, seed=42).to(device)
    perm_layer = ChannelPermute(perm).to(device)

    z = torch.randn(2, 64, 32, 32, device=device)
    z_perm = perm_layer(z)
    z_inv = perm_layer.inverse(z_perm)
    assert torch.allclose(z, z_inv, atol=1e-6), "Phép giải mã ngược ChannelPermute không khớp!"
    print("  -> Kiểm tra ChannelPermute forward/inverse bảo toàn 100%: ĐẠT!")

    # Adapter recovery test trên ma trận giả lập
    adapter = Adapter(channels=64).to(device)
    # Giả lập ma trận hoán vị hoàn hảo
    P_pi_T = torch.zeros(64, 64, device=device)
    for c in range(64):
        P_pi_T[perm[c], c] = 1.0
    adapter.conv.weight.data.copy_(P_pi_T.unsqueeze(-1).unsqueeze(-1))
    hat_pi = recover_perm_from_adapter(adapter.get_matrix())
    acc = match_accuracy(perm, hat_pi)
    assert acc == 1.0, f"Độ khớp phục hồi hoán vị phải là 100%, thực tế: {acc}"
    print("  -> Kiểm tra Cut-Layer Adapter khôi phục hoán vị (100.0%): ĐẠT!")


def test_defenses_step3(device):
    print("\n[TEST 4/5] KIỂM THỬ BASELINES B1 (GAUSSIAN NOISE) VÀ B2 (DP-SGD)...")
    # Gaussian noise
    noise_layer = GaussianNoise(sigma=0.5).to(device)
    z = torch.zeros(1000, 64, 4, 4, device=device)
    z_noisy = noise_layer(z)
    measured_std = z_noisy.std().item()
    assert abs(measured_std - 0.5) < 0.05, f"Độ lệch chuẩn nhiễu không đạt: {measured_std}"
    print(f"  -> Kiểm tra Gaussian Noise (kỳ vọng 0.5, thực tế {measured_std:.4f}): ĐẠT!")

    # DP-SGD
    client = ClientModel().to(device)
    base_opt = torch.optim.SGD(client.parameters(), lr=0.01)
    dp_opt = DPSGDClientOptimizer(client, base_opt, max_grad_norm=1.0, noise_multiplier=1.0)
    x = torch.randn(2, 3, 32, 32, device=device)
    grad_z = torch.randn(2, 64, 32, 32, device=device)
    stats = dp_opt.step(x, grad_z)
    assert "grad_norm" in stats and "noise_norm" in stats
    print("  -> Kiểm tra DPSGDClientOptimizer clipping và noise addition: ĐẠT!")

    eps = compute_dp_epsilon(epochs=10, batch_size=128, dataset_size=50000, noise_multiplier=1.0)
    assert eps > 0.0 and eps < 100.0
    print(f"  -> Kiểm tra RDP Accountant (10 epochs, sigma=1.0 -> Epsilon={eps:.2f}): ĐẠT!")


def test_plotting():
    print("\n[TEST 5/5] KIỂM THỬ CÔNG CỤ VẼ ĐỒ THỊ...")
    dummy_history = [
        {"epoch": 1, "train_loss": 1.5, "train_acc": 0.5, "test_loss": 1.2, "test_acc": 0.6, "lr": 0.1, "epoch_time": 1.0},
        {"epoch": 2, "train_loss": 0.8, "train_acc": 0.7, "test_loss": 0.7, "test_acc": 0.8, "lr": 0.05, "epoch_time": 1.0},
    ]
    test_img = os.path.join(PROJECT_ROOT, "test_dummy_plot.png")
    plot_training_curves(dummy_history, save_path=test_img)
    assert os.path.isfile(test_img), "Không thể lưu file ảnh đồ thị!"
    os.remove(test_img)
    print("  -> Kiểm tra tạo và xuất biểu đồ matplotlib: ĐẠT!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"BẮT ĐẦU CHẠY KIỂM THỬ ĐƠN VỊ TOÀN BỘ HỆ THỐNG ABRETAPE")
    print(f"Thiết bị kiểm thử: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    test_models_and_sl(device)
    test_attacks_and_metrics(device)
    test_defenses_step2(device)
    test_defenses_step3(device)
    test_plotting()

    print("\n" + "=" * 70)
    print(">>> TẤT CẢ CÁC MODULE VÀ THỬ NGHIỆM ĐỀU VƯỢT QUA KIỂM THỬ 100%! <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
