# Bước 3 — Smoke test kiểm tra pipeline Baseline B2 (DP-SGD on Client)
import sys
import os
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP0_DIR = os.path.join(PROJECT_ROOT, "step0")
STEP1_DIR = os.path.join(PROJECT_ROOT, "step1")

for path in reversed([SCRIPT_DIR, STEP0_DIR, STEP1_DIR, PROJECT_ROOT]):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from model import ClientModel, ServerModel
from decoder import Decoder
from metrics import psnr_ssim
from dpsgd import compute_dp_epsilon, DPSGDClientOptimizer
from train_b2 import train_sl_dp_epoch, evaluate_sl_b2
from attack_b2 import train_decoder_b2_epoch, evaluate_attack_b2
from plot_b2 import plot_b2_tradeoff, plot_b2_reconstruction_grid

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def test_dp_epsilon_computation():
    print("[TEST 1/5] Kiểm tra hàm tính toán ngân sách riêng tư RDP (compute_dp_epsilon)...", end=" ")
    eps_05 = compute_dp_epsilon(epochs=100, batch_size=128, dataset_size=50000, noise_multiplier=0.5, delta=1e-5)
    eps_10 = compute_dp_epsilon(epochs=100, batch_size=128, dataset_size=50000, noise_multiplier=1.0, delta=1e-5)
    eps_20 = compute_dp_epsilon(epochs=100, batch_size=128, dataset_size=50000, noise_multiplier=2.0, delta=1e-5)

    assert 0 < eps_20 < eps_10 < eps_05, f"Tính đơn điệu của epsilon không đúng: eps_20={eps_20}, eps_10={eps_10}, eps_05={eps_05}"
    assert eps_10 < 10.0, f"Epsilon ở sigma=1.0 quá lớn: {eps_10}"
    print(f"PASSED! (eps(0.5)={eps_05:.2f}, eps(1.0)={eps_10:.2f}, eps(2.0)={eps_20:.2f})")


def test_dpsgd_clipping_and_noise():
    print("[TEST 2/5] Kiểm tra Gradient Clipping và Noise Injection (DPSGDClientOptimizer)...", end=" ")
    client = ClientModel()
    opt = torch.optim.SGD(client.parameters(), lr=0.01)

    # 1. Kiểm tra Batch-level clipping
    dp_batch = DPSGDClientOptimizer(client, opt, max_grad_norm=1.0, noise_multiplier=1.0, micro_batch_size=None)
    x = torch.randn(8, 3, 32, 32)
    grad_z = torch.randn(8, 64, 32, 32) * 100.0  # Tạo gradient cực lớn để kích hoạt clipping
    stats_batch = dp_client_step_norm(dp_batch, client, x, grad_z)
    assert stats_batch["grad_norm"] > 0, "Grad norm phải dương"

    # 2. Kiểm tra Micro-batch clipping
    dp_micro = DPSGDClientOptimizer(client, opt, max_grad_norm=1.0, noise_multiplier=1.0, micro_batch_size=4)
    stats_micro = dp_micro.step(x, grad_z)
    assert stats_micro["noise_norm"] > 0, "Noise norm phải dương khi noise_multiplier > 0"

    print("PASSED!")


def dp_client_step_norm(dp_client, client, x, grad_z):
    stats = dp_client.step(x, grad_z)
    return stats


def test_sl_dp_training_step():
    print("[TEST 3/5] Kiểm tra 1 epoch huấn luyện Split Learning với Client DP-SGD...", end=" ")
    device = torch.device("cpu")
    client = ClientModel().to(device)
    server = ServerModel().to(device)
    opt_c = torch.optim.SGD(client.parameters(), lr=0.01)
    opt_s = torch.optim.SGD(server.parameters(), lr=0.01)
    dp_client = DPSGDClientOptimizer(client, opt_c, max_grad_norm=1.0, noise_multiplier=0.5, micro_batch_size=4)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    loss_train, acc_train = train_sl_dp_epoch(client, server, dp_client, loader, opt_s, criterion, device)
    assert isinstance(loss_train, float) and loss_train > 0, f"Loss không hợp lệ: {loss_train}"
    assert 0.0 <= acc_train <= 1.0, f"Acc không hợp lệ: {acc_train}"

    val_loss, val_acc = evaluate_sl_b2(client, server, loader, device, criterion=criterion)
    assert 0.0 <= val_acc <= 1.0, f"Val acc không hợp lệ: {val_acc}"
    print(f"PASSED! (Train Loss: {loss_train:.4f}, Train Acc: {acc_train*100:.1f}%)")


def test_decoder_attack_step():
    print("[TEST 4/5] Kiểm tra Decoder tấn công tái tạo trên Client DP-SGD...", end=" ")
    device = torch.device("cpu")
    client = ClientModel().to(device)
    decoder = Decoder().to(device)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.randn(8, 3, 32, 32)
    dataset = TensorDataset(x, torch.zeros(8))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    loss_dec = train_decoder_b2_epoch(client, decoder, loader, opt_d, criterion, device)
    assert isinstance(loss_dec, float) and loss_dec > 0, f"Decoder loss không hợp lệ: {loss_dec}"

    mse, psnr, ssim, lpips = evaluate_attack_b2(client, decoder, loader, device, MEAN, STD, criterion=criterion)
    assert psnr > 0, f"PSNR không hợp lệ: {psnr}"
    assert -1.0 <= ssim <= 1.0, f"SSIM không hợp lệ: {ssim}"
    print(f"PASSED! (Recon MSE: {loss_dec:.4f}, PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})")


def test_b2_end_to_end_and_plot():
    print("[TEST 5/5] Kiểm tra toàn vẹn đồ thị Trade-off và lưới ảnh tái tạo B2...", end=" ")
    temp_dir = tempfile.mkdtemp(prefix="test_b2_")
    try:
        results = [
            {"sigma_dp": 0.5, "epsilon": 25.2, "clip_norm": 1.0, "test_acc": 0.88, "mse": 0.015, "psnr": 22.5, "ssim": 0.75, "lpips": None},
            {"sigma_dp": 1.0, "epsilon": 3.41, "clip_norm": 1.0, "test_acc": 0.81, "mse": 0.018, "psnr": 21.0, "ssim": 0.71, "lpips": None},
            {"sigma_dp": 2.0, "epsilon": 1.33, "clip_norm": 1.0, "test_acc": 0.70, "mse": 0.021, "psnr": 19.8, "ssim": 0.67, "lpips": None},
        ]
        tradeoff_path = os.path.join(temp_dir, "test_tradeoff.png")
        plot_b2_tradeoff(results, save_path=tradeoff_path)
        assert os.path.exists(tradeoff_path), "File đồ thị trade-off không được tạo"

        dummy_orig = torch.randn(6, 3, 32, 32)
        dummy_recons = {
            0.5: torch.randn(6, 3, 32, 32),
            1.0: torch.randn(6, 3, 32, 32),
            2.0: torch.randn(6, 3, 32, 32),
        }
        dummy_eps = {0.5: 25.2, 1.0: 3.41, 2.0: 1.33}
        grid_path = os.path.join(temp_dir, "test_grid.png")
        plot_b2_reconstruction_grid(dummy_orig, dummy_recons, eps_by_sigma=dummy_eps, mean=MEAN, std=STD, save_path=grid_path)
        assert os.path.exists(grid_path), "File lưới ảnh tái tạo không được tạo"

        print("PASSED!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("\n==================================================")
    print("   BẮT ĐẦU KIỂM THỬ KHẢ NĂNG HOẠT ĐỘNG BASELINE B2")
    print("==================================================")
    test_dp_epsilon_computation()
    test_dpsgd_clipping_and_noise()
    test_sl_dp_training_step()
    test_decoder_attack_step()
    test_b2_end_to_end_and_plot()
    print("==================================================")
    print("   TẤT CẢ 5/5 TESTS CỦA BASELINE B2 ĐÃ THÀNH CÔNG!   ")
    print("==================================================\n")
