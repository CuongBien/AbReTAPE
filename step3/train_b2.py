# Bước 3 — Vòng lặp huấn luyện Split Learning có bảo vệ DP-SGD trên Client
import torch
import torch.nn as nn


def train_sl_dp_epoch(client, server, dp_client, loader, opt_s, criterion, device):
    """
    Huấn luyện 1 epoch Split Learning với Client cập nhật theo cơ chế DP-SGD:
    Forward:  x -> client -> z -> server -> logits
    Backward: server -> dL/dz (gửi về Client)
              client -> DP-SGD step (clip gradient + cộng nhiễu Gauss)
    
    Tham số:
        client: ClientModel
        server: ServerModel
        dp_client: DPSGDClientOptimizer
        loader: DataLoader tập train
        opt_s: Optimizer của Server (SGD/Adam)
        criterion: Loss function (CrossEntropyLoss)
        device: torch.device
    """
    client.train()
    server.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        batch_size = x.size(0)

        opt_s.zero_grad()
        dp_client.zero_grad()

        # 1. Client forward: sinh ra smashed data z
        z = client(x)

        # 2. Cắt đồ thị tính toán tại cut layer
        z_d = z.detach().requires_grad_(True)

        # 3. Server forward và tính loss tác vụ
        logits = server(z_d)
        loss = criterion(logits, y)

        # 4. Server backward và cập nhật trọng số Server
        loss.backward()
        opt_s.step()

        # 5. Client nhận dL/dz và thực hiện DP-SGD step
        grad_z = z_d.grad
        dp_client.step(x, grad_z)

        total_loss += loss.item() * batch_size
        correct += (logits.argmax(1) == y).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_sl_b2(client, server, loader, device, criterion=None):
    """
    Đánh giá độ chính xác phân loại (Utility) của hệ thống Split Learning (B2).
    """
    client.eval()
    server.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        batch_size = x.size(0)

        z = client(x)
        logits = server(z)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * batch_size

        correct += (logits.argmax(1) == y).sum().item()
        total += batch_size

    val_loss = (total_loss / total) if criterion is not None else 0.0
    val_acc = correct / total
    return val_loss, val_acc
