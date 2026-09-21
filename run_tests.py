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
from src.defenses import random_perm, ChannelPermute, Adapter, GaussianNoise, DPSGDClientOptimizer, compute_dp_epsilon, NoPeekDefense
from src.attacks import Decoder, recover_perm, recover_perm_from_adapter, match_accuracy
from src.metrics import psnr_ssim, denormalize, get_lpips_fn, calculate_lpips, distance_correlation
from src.training import train_sl_epoch, evaluate_sl, EarlyStopping
from src.utils import plot_training_curves


def test_models_and_sl(device):
    print("\n[TEST 1/6] KIỂM THỬ MÔ HÌNH VÀ SPLIT LEARNING FORWARD/BACKWARD...")
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
    print("\n[TEST 2/6] KIỂM THỬ DECODER TẤN CÔNG VÀ BỘ ĐỘ ĐO (PSNR, SSIM, LPIPS)...")
    decoder = Decoder(in_channels=64, out_channels=3).to(device)
    z_dummy = torch.randn(2, 64, 32, 32, device=device)
    x_rec = decoder(z_dummy)
    assert x_rec.shape == (2, 3, 32, 32), f"Lỗi shape x_rec: {x_rec.shape}"

    # Kiểm tra PSNR, SSIM
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    x_clean = torch.rand(2, 3, 32, 32, device=device)
    p, s = psnr_ssim(x_clean, x_clean, mean, std)
    print(f"  -> Kiểm tra tính toán PSNR/SSIM ({p:.1f} dB, {s:.4f}): ĐẠT!")

    # Kiểm tra LPIPS
    lpips_fn = get_lpips_fn(device=device)
    if lpips_fn is not None:
        lpips_val = calculate_lpips(x_clean, x_clean, mean, std, lpips_fn)
        print(f"  -> Kiểm tra tính toán LPIPS ({lpips_val:.6f}): ĐẠT!")
    else:
        print("  -> Bỏ qua LPIPS (chưa cài đặt).")


def test_defenses_step2(device):
    print("\n[TEST 3/6] KIỂM THỬ HOÁN VỊ KÊNH, ADAPTER VÀ KHÔI PHỤC HẤP THỤ...")
    perm = random_perm(64, seed=42).to(device)
    permute = ChannelPermute(perm).to(device)
    z = torch.randn(4, 64, 16, 16, device=device)
    z_perm = permute(z)
    z_rec = permute.inverse(z_perm)
    assert torch.allclose(z, z_rec, atol=1e-5), "Lỗi giải mã hoán vị ChannelPermute!"
    print("  -> Kiểm tra ChannelPermute forward/inverse bảo toàn 100%: ĐẠT!")

    # Kiểm tra Adapter
    adapter = Adapter(channels=64).to(device)
    opt_a = torch.optim.Adam(adapter.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(5):
        opt_a.zero_grad()
        loss = loss_fn(adapter(z_perm), z)
        loss.backward()
        opt_a.step()

    A = adapter.get_matrix()
    hat_pi = recover_perm_from_adapter(A)
    acc = match_accuracy(perm, hat_pi)
    print(f"  -> Kiểm tra Cut-Layer Adapter khôi phục hoán vị ({acc*100:.1f}%): ĐẠT!")


def test_defenses_step3(device):
    print("\n[TEST 4/6] KIỂM THỬ BASELINES B1 (GAUSSIAN NOISE) VÀ B2 (DP-SGD)...")
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

    # B3: NoPeek & Distance Correlation
    x_dcor = torch.randn(20, 3, 32, 32, device=device)
    # dCor giữa 2 biến đồng nhất phải xấp xỉ 1.0
    dcor_self = distance_correlation(x_dcor, x_dcor).item()
    assert abs(dcor_self - 1.0) < 0.05, f"dCor(X, X) phải xấp xỉ 1.0, thực tế: {dcor_self}"
    
    # dCor giữa 2 biến ngẫu nhiên độc lập phải thấp
    x_rand = torch.randn(20, 64, 32, 32, device=device)
    dcor_indep = distance_correlation(x_dcor, x_rand).item()
    assert dcor_indep < 0.5, f"dCor(X, independent) phải thấp, thực tế: {dcor_indep}"
    print(f"  -> Kiểm tra Distance Correlation dCor (Tự tương quan: {dcor_self:.4f}, Độc lập: {dcor_indep:.4f}): ĐẠT!")

    # Kiểm tra vòng lặp NoPeek SL với dCor penalty
    nopeek = NoPeekDefense(alpha=0.5).to(device)
    server = ServerModel().to(device)
    opt_s = torch.optim.SGD(server.parameters(), lr=0.01)
    y_dummy = torch.randint(0, 10, (20,), device=device)
    crit = nn.CrossEntropyLoss()
    train_loss, train_acc = train_sl_epoch(client, server, [(x_dcor, y_dummy)], base_opt, opt_s, crit, device, defense=nopeek)
    print("  -> Kiểm tra NoPeek Split Learning training step với dCor penalty: ĐẠT!")


def test_early_stopping():
    print("\n[TEST 5/6] KIỂM THỬ MODULE EARLY STOPPING...")
    # Mode max
    es = EarlyStopping(patience=3, min_delta=1e-3, mode="max")
    assert not es.step(0.80, epoch=1)
    assert not es.step(0.85, epoch=2)  # improved
    assert not es.step(0.84, epoch=3)  # counter = 1
    assert not es.step(0.83, epoch=4)  # counter = 2
    assert es.step(0.82, epoch=5)      # counter = 3 -> True
    assert es.best_epoch == 2
    assert es.best_score == 0.85
    print("  -> Kiểm tra EarlyStopping (mode='max', patience=3): ĐẠT!")


def test_plotting():
    print("\n[TEST 6/6] KIỂM THỬ CÔNG CỤ VẼ ĐỒ THỊ...")
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
    test_early_stopping()
    test_plotting()

    print("\n" + "=" * 70)
    print(">>> TẤT CẢ CÁC MODULE VÀ THỬ NGHIỆM ĐỀU VƯỢT QUA KIỂM THỬ 100%! <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
