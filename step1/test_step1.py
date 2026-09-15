# Kiểm thử đơn vị (Smoke Test) cho Bước 1 — Passive Reconstruction Attack
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

for path in [SCRIPT_DIR, PROJECT_ROOT, STEP0_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model import ClientModel
from decoder import Decoder
from attack import train_decoder_epoch, evaluate_attack
from metrics import denormalize, psnr_ssim
from plot_attack import save_reconstruction_grid, plot_attack_curves

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def test_step1_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TEST BƯỚC 1] Thiết bị kiểm thử: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Kiểm tra kích thước mạng Decoder
    decoder = Decoder().to(device)
    dummy_z = torch.randn(2, 64, 32, 32, device=device)
    x_hat = decoder(dummy_z)
    assert x_hat.shape == (2, 3, 32, 32), f"Sai kích thước x_hat: {x_hat.shape}, mong đợi (2, 3, 32, 32)"
    print(f"[TEST] Decoder forward pass thành công! Shape IR z: {list(dummy_z.shape)} -> Shape x_hat: {list(x_hat.shape)}")

    # 2. Kiểm tra Client F_c bị đóng băng hoàn toàn
    client = ClientModel().to(device)
    client.eval()
    for p in client.parameters():
        p.requires_grad = False

    dummy_x = torch.randn(8, 3, 32, 32)
    dummy_y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    w_client_before = next(client.parameters()).clone().detach()
    w_dec_before = next(decoder.parameters()).clone().detach()

    train_mse = train_decoder_epoch(client, decoder, loader, opt, criterion, device)
    print(f"[TEST] Train Decoder MSE 1 epoch: {train_mse:.4f}")

    w_client_after = next(client.parameters()).clone().detach()
    w_dec_after = next(decoder.parameters()).clone().detach()

    assert torch.equal(w_client_before, w_client_after), "LỖI: Client weights đã bị thay đổi (Client không được đóng băng)!"
    assert not torch.equal(w_dec_before, w_dec_after), "LỖI: Decoder weights không được cập nhật sau backpropagation!"
    print("[TEST] Client đóng băng hoàn toàn và Decoder cập nhật trọng số chính xác!")

    # 3. Kiểm tra đo đạc PSNR và SSIM
    test_mse, test_psnr, test_ssim, test_lpips = evaluate_attack(
        client, decoder, loader, device, MEAN, STD, criterion=criterion
    )
    print(f"[TEST] Evaluate test_mse: {test_mse:.4f} | PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f}")
    assert test_psnr > 0, "LỖI: Giá trị PSNR không hợp lệ!"
    assert 0.0 <= test_ssim <= 1.0, "LỖI: Giá trị SSIM ngoài khoảng [0, 1]!"

    # 4. Kiểm tra xuất lưới ảnh và đồ thị
    dummy_grid_file = os.path.join(SCRIPT_DIR, "test_dummy_grid.png")
    dummy_curve_file = os.path.join(SCRIPT_DIR, "test_dummy_curves.png")

    save_reconstruction_grid(client, decoder, loader, device, MEAN, STD, save_path=dummy_grid_file, num_images=4)
    assert os.path.isfile(dummy_grid_file), "LỖI: Không tạo được ảnh lưới đối chứng!"
    os.remove(dummy_grid_file)

    dummy_hist = [
        {"epoch": 1, "train_mse": 0.5, "test_mse": 0.45, "test_psnr": 15.0, "test_ssim": 0.4, "test_lpips": 0.3, "epoch_time": 1.0},
        {"epoch": 2, "train_mse": 0.2, "test_mse": 0.18, "test_psnr": 22.0, "test_ssim": 0.7, "test_lpips": 0.15, "epoch_time": 1.0},
    ]
    plot_attack_curves(dummy_hist, save_path=dummy_curve_file, show=False)
    assert os.path.isfile(dummy_curve_file), "LỖI: Không tạo được đồ thị tấn công!"
    os.remove(dummy_curve_file)
    print("[TEST] Module vẽ đồ thị và xuất lưới ảnh hoạt động hoàn hảo!")

    print("\n>>> TẤT CẢ CÁC KIỂM THỬ ĐƠN VỊ BƯỚC 1 ĐỀU ĐẠT CHUẨN! <<<")


if __name__ == "__main__":
    test_step1_pipeline()
