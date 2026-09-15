# Kiểm thử đơn vị (Smoke Test) cho Bước 3 — Baseline 1: Nhiễu Gaussian
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Thiết lập UTF-8 encoding cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
from defense import GaussianNoise
from train_sl import train_epoch, evaluate_sl
from attack import train_adaptive_decoder_epoch, evaluate_attack
from plot_b1 import plot_tradeoff, plot_reconstruction_grid

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def test_step3_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TEST BƯỚC 3] Thiết bị kiểm thử: {device}")

    # 1. Kiểm tra GaussianNoise và phương sai nhiễu
    sigma = 0.5
    noise_layer = GaussianNoise(sigma=sigma, apply_on_eval=True).to(device)
    dummy_z = torch.zeros(1000, 64, 8, 8, device=device)
    noisy_z = noise_layer(dummy_z)

    assert noisy_z.shape == dummy_z.shape, f"Sai kích thước z': {noisy_z.shape}"
    emp_std = noisy_z.std().item()
    print(f"[TEST] GaussianNoise: Kỳ vọng σ = {sigma}, Thực tế đo được = {emp_std:.4f}")
    assert abs(emp_std - sigma) < 0.05, f"Phương sai nhiễu thực nghiệm lệch quá nhiều: {emp_std} vs {sigma}"
    print("[TEST] GaussianNoise hoạt động chuẩn xác 100%!")

    # 2. Kiểm tra Gradient flow trong Split Learning có phòng thủ
    client = ClientModel().to(device)
    server = ServerModel().to(device)

    opt_c = torch.optim.SGD(client.parameters(), lr=0.1)
    opt_s = torch.optim.SGD(server.parameters(), lr=0.1)
    criterion_task = nn.CrossEntropyLoss()

    dummy_x = torch.randn(8, 3, 32, 32)
    dummy_y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=4, shuffle=False)

    w_c_before = next(client.parameters()).clone().detach()
    w_s_before = next(server.parameters()).clone().detach()

    train_loss, train_acc = train_epoch(client, server, noise_layer, loader, opt_c, opt_s, criterion_task, device)
    print(f"[TEST] Train SL 1 epoch (σ={sigma}): Loss = {train_loss:.4f}, Acc = {train_acc*100:.2f}%")

    w_c_after = next(client.parameters()).clone().detach()
    w_s_after = next(server.parameters()).clone().detach()

    assert not torch.equal(w_c_before, w_c_after), "LỖI: Gradient không truyền qua nhiễu về Client!"
    assert not torch.equal(w_s_before, w_s_after), "LỖI: Trọng số Server không thay đổi!"
    print("[TEST] Gradient lan truyền ngược qua nhiễu Gaussian thành công!")

    # 3. Kiểm tra Pha Tấn công Tái tạo Thích ứng (Decoder Attack)
    decoder = Decoder().to(device)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion_recon = nn.MSELoss()

    w_d_before = next(decoder.parameters()).clone().detach()
    dec_loss = train_adaptive_decoder_epoch(client, decoder, noise_layer, loader, opt_d, criterion_recon, device)
    w_d_after = next(decoder.parameters()).clone().detach()

    print(f"[TEST] Train Decoder 1 epoch: Loss (MSE) = {dec_loss:.5f}")
    assert not torch.equal(w_d_before, w_d_after), "LỖI: Trọng số Decoder không cập nhật!"

    # 4. Kiểm tra hàm đánh giá Security (PSNR, SSIM)
    mean_mse, mean_psnr, mean_ssim, mean_lpips = evaluate_attack(
        client, decoder, noise_layer, loader, device, MEAN, STD, criterion=criterion_recon, lpips_fn=None
    )
    print(f"[TEST] Đánh giá Security: MSE = {mean_mse:.5f}, PSNR = {mean_psnr:.2f} dB, SSIM = {mean_ssim:.4f}")
    assert mean_psnr > 0.0, "PSNR phải là số dương hợp lệ!"
    assert 0.0 <= mean_ssim <= 1.0, "SSIM phải nằm trong khoảng [0, 1]!"

    # 5. Kiểm tra xuất đồ thị Trade-off và lưới ảnh tái tạo
    dummy_results = [
        {"sigma": 0.1, "test_acc": 0.935, "mse": 0.015, "psnr": 26.5, "ssim": 0.88, "lpips": 0.12},
        {"sigma": 0.5, "test_acc": 0.902, "mse": 0.045, "psnr": 19.8, "ssim": 0.65, "lpips": 0.28},
        {"sigma": 1.0, "test_acc": 0.824, "mse": 0.095, "psnr": 15.2, "ssim": 0.42, "lpips": 0.45},
    ]

    test_plot_path = os.path.join(SCRIPT_DIR, "test_tradeoff.png")
    plot_tradeoff(dummy_results, save_path=test_plot_path, show=False)
    assert os.path.isfile(test_plot_path), "LỖI: Không tạo được đồ thị Trade-off!"
    os.remove(test_plot_path)

    test_grid_path = os.path.join(SCRIPT_DIR, "test_grid.png")
    dummy_recons = {
        0.1: torch.randn(4, 3, 32, 32),
        0.5: torch.randn(4, 3, 32, 32),
        1.0: torch.randn(4, 3, 32, 32),
    }
    plot_reconstruction_grid(dummy_x[:4], dummy_recons, MEAN, STD, save_path=test_grid_path, num_images=4, show=False)
    assert os.path.isfile(test_grid_path), "LỖI: Không tạo được lưới ảnh tái tạo!"
    os.remove(test_grid_path)

    print("[TEST] Module vẽ đồ thị Trade-off và lưới ảnh tái tạo hoạt động hoàn hảo!")
    print("\n>>> TẤT CẢ CÁC KIỂM THỬ ĐƠN VỊ BƯỚC 3 ĐỀU ĐẠT CHUẨN 100%! <<<")


if __name__ == "__main__":
    test_step3_pipeline()
