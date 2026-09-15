# Bước 0 — vòng lặp huấn luyện Split Learning (đơn tiến trình)
import torch
import torch.nn as nn


def train_epoch(client, server, loader, opt_c, opt_s, criterion, device):
    client.train()
    server.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt_c.zero_grad()
        opt_s.zero_grad()

        # Forward: client -> smashed data z -> server
        z = client(x)  # IR, giữ đồ thị để backprop client
        z_d = z.detach().requires_grad_(True)  # thứ server nhận (đã cắt đồ thị)
        logits = server(z_d)
        loss = criterion(logits, y)

        # Backward phía server
        loss.backward()  # tính z_d.grad = dL/dz
        opt_s.step()

        # Backward phía client: chỉ dL/dz đi ngược qua biên giới
        z.backward(z_d.grad)  # dL/dtheta_c
        opt_c.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(client, server, loader, device):
    client.eval()
    server.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = server(client(x))
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total
