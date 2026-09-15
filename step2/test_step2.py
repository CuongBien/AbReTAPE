# Kiểm thử đơn vị (Smoke Test) cho Bước 2 — Thí nghiệm Hấp thụ
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

from model import ClientModel, ServerModel
from permute import random_perm, ChannelPermute
from train_sl import train_epoch, evaluate_sl
from recover import recover_perm, match_accuracy
from plot_absorption import plot_permutation_matrix, plot_step2_curves


def test_step2_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TEST BƯỚC 2] Thiết bị kiểm thử: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Kiểm tra ChannelPermute và tính khả nghịch
    perm = random_perm(64, seed=42)
    permute = ChannelPermute(perm).to(device)

    dummy_z = torch.randn(4, 64, 32, 32, device=device)
    z_prime = permute(dummy_z)
    assert z_prime.shape == (4, 64, 32, 32), f"Sai kích thước z': {z_prime.shape}"

    z_rec = permute.inverse(z_prime)
    assert torch.allclose(dummy_z, z_rec, atol=1e-6), "LỖI: Phép giải mã ngược lý thuyết inverse() không khớp với z gốc!"
    print("[TEST] ChannelPermute hoạt động chính xác và bảo toàn thông tin 100%!")

    # 2. Kiểm tra lan truyền xuôi và ngược qua biên giới hoán vị
    client = ClientModel().to(device)
    server = ServerModel().to(device)

    opt_c = torch.optim.SGD(client.parameters(), lr=0.1)
    opt_s = torch.optim.SGD(server.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    dummy_x = torch.randn(8, 3, 32, 32)
    dummy_y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    w_c_before = next(client.parameters()).clone().detach()
    w_s_before = next(server.parameters()).clone().detach()

    train_loss, train_acc = train_epoch(client, server, permute, loader, opt_c, opt_s, criterion, device)
    print(f"[TEST] Train SL với hoán vị 1 epoch: Loss = {train_loss:.4f}, Acc = {train_acc*100:.2f}%")

    w_c_after = next(client.parameters()).clone().detach()
    w_s_after = next(server.parameters()).clone().detach()

    assert not torch.equal(w_c_before, w_c_after), "LỖI: Trọng số Client không thay đổi sau train_epoch!"
    assert not torch.equal(w_s_before, w_s_after), "LỖI: Trọng số Server không thay đổi sau train_epoch!"
    print("[TEST] Gradient lan truyền ngược qua biên giới hoán vị thành công tới cả 2 phía!")

    # 2b. Kiểm tra chế độ Đóng băng Client (opt_c=None) chuẩn cho thí nghiệm hấp thụ
    w_c_frozen = next(client.parameters()).clone().detach()
    train_loss_frz, train_acc_frz = train_epoch(client, server, permute, loader, None, opt_s, criterion, device)
    w_c_frozen_after = next(client.parameters()).clone().detach()
    assert torch.equal(w_c_frozen, w_c_frozen_after), "LỖI: Trọng số Client thay đổi dù đã đóng băng (opt_c=None)!"
    print("[TEST] Chế độ Đóng băng Client (Frozen Client) hoạt động chuẩn xác 100%!")

    # 3. Kiểm chứng thuật toán khôi phục hoán vị ( recover_perm )
    # Tạo trọng số tham chiếu W_ref ngẫu nhiên [128, 64, 3, 3]
    torch.manual_seed(123)
    W_ref = torch.randn(128, 64, 3, 3)

    # Giả lập Server đã hấp thụ hoán vị: W[:, c] = W_ref[:, perm[c]]
    W_simulated = W_ref[:, perm, :, :].clone()
    # Thêm một chút nhiễu nhỏ để kiểm tra tính bền vững
    W_simulated += torch.randn_like(W_simulated) * 1e-4

    perm_recovered = recover_perm(W_simulated, W_ref)
    acc = match_accuracy(perm, perm_recovered)
    print(f"[TEST] Khôi phục hoán vị trên ma trận giả lập: Độ khớp = {acc*100:.1f}%")
    assert acc == 1.0, f"LỖI: Thuật toán khôi phục không đạt 100% trên dữ liệu lý tưởng (đạt {acc*100:.1f}%)!"

    # Đối chứng với ngẫu nhiên
    rand_perm = torch.randperm(64).tolist()
    rand_acc = match_accuracy(perm, rand_perm)
    print(f"[TEST] Đối chứng ngẫu nhiên: {rand_acc*100:.2f}% (kỳ vọng ~1.56%)")

    # 4. Kiểm tra module xuất Heatmap và đồ thị
    dummy_heatmap_file = os.path.join(SCRIPT_DIR, "test_dummy_heatmap.png")
    dummy_curves_file = os.path.join(SCRIPT_DIR, "test_dummy_curves.png")

    plot_permutation_matrix(W_simulated, W_ref, perm, save_path=dummy_heatmap_file, show=False)
    assert os.path.isfile(dummy_heatmap_file), "LỖI: Không tạo được file Heatmap!"
    os.remove(dummy_heatmap_file)

    dummy_history = [
        {"epoch": 1, "train_loss": 2.3, "train_acc": 0.15, "test_loss": 2.1, "test_acc": 0.20, "match_acc": 0.10, "rand_acc": 0.015, "lr": 0.1, "epoch_time": 1.0},
        {"epoch": 10, "train_loss": 0.5, "train_acc": 0.85, "test_loss": 0.6, "test_acc": 0.82, "match_acc": 0.95, "rand_acc": 0.015, "lr": 0.01, "epoch_time": 1.0},
    ]
    plot_step2_curves(dummy_history, save_path=dummy_curves_file, show=False)
    assert os.path.isfile(dummy_curves_file), "LỖI: Không tạo được file đồ thị huấn luyện!"
    os.remove(dummy_curves_file)
    # 5. Kiểm thử Cut-Layer Adapter (Cách 3)
    from adapter import Adapter
    from recover import recover_perm_from_adapter, recover_perm_covariance

    adapter = Adapter(64).to(device)
    dummy_z_perm = permute(dummy_z)
    z_hat = adapter(dummy_z_perm)
    assert z_hat.shape == (4, 64, 32, 32), f"Sai kích thước output của Adapter: {z_hat.shape}"

    # Giả lập ma trận Adapter đã hội tụ về P_pi^T
    P_pi_T = torch.zeros(64, 64)
    for c in range(64):
        P_pi_T[perm[c], c] = 1.0  # Hàng perm[c], cột c = 1.0
    perm_from_adp = recover_perm_from_adapter(P_pi_T)
    acc_adp = match_accuracy(perm, perm_from_adp)
    assert acc_adp == 1.0, f"LỖI: recover_perm_from_adapter không đạt 100% trên ma trận lý tưởng ({acc_adp*100:.1f}%)!"
    print("[TEST] Cut-Layer Adapter và thuật toán khôi phục recover_perm_from_adapter đạt chuẩn 100%!")

    # 6. Kiểm thử Channel Covariance Recovery (Cách 2)
    # Giả lập client trả về z có phương sai khác biệt giữa các kênh
    class DummyClient(nn.Module):
        def forward(self, x):
            # Mỗi kênh nhân với một scale khác nhau
            scales = torch.linspace(0.1, 10.0, 64, device=x.device).view(1, 64, 1, 1)
            return torch.randn(x.size(0), 64, 8, 8, device=x.device) * scales

    dummy_client = DummyClient().to(device)
    cov_perm_hat, cov_acc = recover_perm_covariance(dummy_client, loader, perm, device, num_batches=2)
    print(f"[TEST] Thống kê Phương sai Kênh (Channel Covariance): Độ khớp = {cov_acc*100:.1f}%")
    assert cov_acc > 0.80, f"LỖI: Covariance recovery không đạt yêu cầu ({cov_acc*100:.1f}%)!"
    print("[TEST] Cơ chế Channel Covariance Statistical Attack hoạt động chính xác!")

    print("\n>>> TẤT CẢ CÁC KIỂM THỬ ĐƠN VỊ BƯỚC 2 ĐỀU ĐẠT CHUẨN! <<<")


if __name__ == "__main__":
    test_step2_pipeline()
