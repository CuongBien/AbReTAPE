# Động cơ huấn luyện Split Learning (đơn tiến trình) hỗ trợ tùy biến các cơ chế phòng thủ
import torch
import torch.nn as nn


def train_sl_epoch(client, server, loader, opt_c, opt_s, criterion, device, defense=None, client_opt_is_dpsgd=False):
    """
    Thực hiện 1 epoch huấn luyện Split Learning 2 bên (Client - Server):
    1. Client forward: z = client(x)
    2. Defense transform (nếu có): z_trans = defense(z)
    3. Cắt đồ thị tại cut-layer: z_d = z_trans.detach().requires_grad_(True)
    4. Server forward: logits = server(z_d)
    5. Server backward: loss.backward() -> tính grad_boundary = z_d.grad
    6. Client backward: grad_boundary truyền ngược qua defense về client
    """
    client.train()
    server.train()
    if defense is not None:
        defense.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        batch_size = x.size(0)

        opt_s.zero_grad()
        if not client_opt_is_dpsgd:
            opt_c.zero_grad()

        # Forward
        z = client(x)
        if defense is not None:
            z_trans = defense(z)
            z_d = z_trans.detach().requires_grad_(True)
        else:
            z_d = z.detach().requires_grad_(True)

        logits = server(z_d)
        loss = criterion(logits, y)

        # Backward Server
        loss.backward()
        opt_s.step()

        # Backward Client
        grad_boundary = z_d.grad
        if client_opt_is_dpsgd:
            # DP-SGD quản lý clipping và backward bên trong opt_c.step
            opt_c.step(x, grad_boundary)
        else:
            if defense is not None:
                z_trans.backward(grad_boundary)
            else:
                z.backward(grad_boundary)
            opt_c.step()

        total_loss += loss.item() * batch_size
        correct += (logits.argmax(1) == y).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_sl(client, server, loader, device, defense=None, criterion=None):
    """
    Đánh giá độ chính xác (Accuracy) và hàm mất mát (Loss) của hệ thống Split Learning trên tập test.
    """
    client.eval()
    server.eval()
    if defense is not None:
        defense.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        batch_size = x.size(0)

        z = client(x)
        if defense is not None:
            z = defense(z)
        logits = server(z)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * batch_size

        correct += (logits.argmax(1) == y).sum().item()
        total += batch_size

    val_loss = (total_loss / total) if criterion is not None else 0.0
    val_acc = correct / total
    return val_loss, val_acc


# Aliases tương thích ngược
train_epoch = train_sl_epoch
evaluate = evaluate_sl
