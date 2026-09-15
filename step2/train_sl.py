# Bước 2 — Vòng lặp huấn luyện Split Learning có chèn hoán vị kênh tại cut layer
import torch
import torch.nn as nn


def train_epoch(client, server, permute, loader, opt_c, opt_s, criterion, device):
    """
    Huấn luyện 1 epoch Split Learning với hoán vị kênh E(z) tại biên giới:
    Forward:  x -> client -> z -> permute(z) -> z' -> server -> logits
    Backward: server -> dL/dz' -> permute.backward -> dL/dz -> client
    """
    if opt_c is not None:
        client.train()
    else:
        client.eval()
    server.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        batch_size = x.size(0)

        opt_s.zero_grad()
        if opt_c is not None:
            opt_c.zero_grad()

        # 1. Client tính biểu diễn trung gian z (IR)
        if opt_c is None:
            with torch.no_grad():
                z = client(x)
        else:
            z = client(x)

        # 2. Chèn hoán vị kênh cố định: z' = E(z)
        z_perm = permute(z)

        # 3. Cắt đồ thị phía Server
        z_d = z_perm.detach().requires_grad_(True)

        # 4. Server tính logits và loss
        logits = server(z_d)
        loss = criterion(logits, y)

        # 5. Lan truyền ngược phía Server
        loss.backward()
        opt_s.step()

        # 6. Lan truyền ngược qua biên giới hoán vị về Client (nếu Client không bị đóng băng)
        if opt_c is not None:
            z_perm.backward(z_d.grad)
            opt_c.step()


        total_loss += loss.item() * batch_size
        correct += (logits.argmax(1) == y).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_sl(client, server, permute, loader, device, criterion=None):
    """
    Đánh giá độ chính xác phân loại của hệ thống Split Learning có hoán vị kênh.
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
        z_perm = permute(z)
        logits = server(z_perm)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * batch_size

        correct += (logits.argmax(1) == y).sum().item()
        total += batch_size

    val_loss = (total_loss / total) if criterion is not None else 0.0
    val_acc = correct / total
    return val_loss, val_acc
