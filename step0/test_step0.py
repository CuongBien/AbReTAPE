# Kiểm thử đơn vị (Smoke Test) cho Bước 0
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Thiết lập UTF-8 encoding cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import ClientModel, ServerModel
from train import train_epoch, evaluate


def test_split_learning_step():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[TEST] Thiết bị kiểm thử: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    client = ClientModel().to(device)
    server = ServerModel().to(device)

    opt_c = torch.optim.SGD(client.parameters(), lr=0.1, momentum=0.9)
    opt_s = torch.optim.SGD(server.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Dữ liệu giả lập 16 ảnh 3x32x32 với 10 nhãn lớp
    dummy_x = torch.randn(16, 3, 32, 32)
    dummy_y = torch.randint(0, 10, (16,))
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Ghi lại trọng số ban đầu để kiểm tra cập nhật gradient
    w_c_before = next(client.parameters()).clone().detach()
    w_s_before = next(server.parameters()).clone().detach()

    # Chạy 1 epoch train
    loss, train_acc = train_epoch(client, server, loader, opt_c, opt_s, criterion, device)
    print(f"[TEST] Train loss: {loss:.4f} | Train acc: {train_acc * 100:.2f}%")

    w_c_after = next(client.parameters()).clone().detach()
    w_s_after = next(server.parameters()).clone().detach()

    # Kiểm tra gradient đã cập nhật cả 2 phía
    client_updated = not torch.equal(w_c_before, w_c_after)
    server_updated = not torch.equal(w_s_before, w_s_after)

    assert client_updated, "Trọng số Client không thay đổi sau train_epoch!"
    assert server_updated, "Trọng số Server không thay đổi sau train_epoch!"
    print("[TEST] Client & Server weights cập nhật thành công qua biên giới!")

    # Chạy evaluate
    val_loss, val_acc = evaluate(client, server, loader, device, criterion=criterion)
    print(f"[TEST] Evaluate loss: {val_loss:.4f} | acc: {val_acc * 100:.2f}%")

    # Kiểm tra kích thước smashed data z
    x_test = torch.randn(2, 3, 32, 32, device=device)
    z = client(x_test)
    assert z.shape == (2, 64, 32, 32), f"Sai kích thước z: {z.shape}, mong đợi (2, 64, 32, 32)"
    logits = server(z)
    assert logits.shape == (2, 10), f"Sai kích thước logits: {logits.shape}, mong đợi (2, 10)"
    print(f"[TEST] Kích thước IR z = {list(z.shape)} và logits = {list(logits.shape)} hoàn toàn chính xác!")

    # Kiểm tra module plot.py
    from plot import plot_history
    dummy_history = [
        {"epoch": 1, "train_loss": 2.3, "train_acc": 0.15, "test_loss": 2.1, "test_acc": 0.20, "lr": 0.1, "epoch_time": 1.2},
        {"epoch": 2, "train_loss": 1.9, "train_acc": 0.32, "test_loss": 1.8, "test_acc": 0.35, "lr": 0.1, "epoch_time": 1.1},
    ]
    test_plot_file = os.path.join(SCRIPT_DIR, "test_dummy_plot.png")
    plot_history(dummy_history, save_path=test_plot_file, show=False)
    assert os.path.isfile(test_plot_file), "File đồ thị test không được tạo thành công!"
    os.remove(test_plot_file)
    print("[TEST] Module vẽ đồ thị plot_history hoạt động hoàn hảo!")

    print("\n>>> TẤT CẢ CÁC KIỂM THỬ BƯỚC 0 ĐỀU ĐẠT CHUẨN! <<<")


if __name__ == '__main__':
    test_split_learning_step()
