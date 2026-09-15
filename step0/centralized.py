# Bước 0 — Centralized Training tham chiếu trên CIFAR-10 (ResNet-18)
import os
import sys
import argparse

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

import torch
import torch.nn as nn
from model import resnet18_cifar
from data import get_cifar10


def parse_args():
    parser = argparse.ArgumentParser(description="Centralized Training on CIFAR-10 (ResNet-18)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (mặc định: 0.1)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (mặc định: 2)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(SCRIPT_DIR, "data"), help="Thư mục chứa CIFAR-10")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "b0_centralized.pt"), help="File lưu checkpoint")
    return parser.parse_args()


def train_epoch_centralized(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_centralized(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("==================================================")
    print(f"Chế độ           : Centralized Training (Tham chiếu)")
    print(f"Thiết bị         : {device}")
    if device == 'cuda':
        print(f"GPU Name         : {torch.cuda.get_device_name(0)}")
    print(f"Epochs           : {args.epochs}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Learning rate    : {args.lr}")
    print(f"Output           : {args.output}")
    print("==================================================")

    torch.manual_seed(0)
    trainloader, testloader = get_cifar10(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = resnet18_cifar().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    print("Bắt đầu huấn luyện Centralized...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch_centralized(model, trainloader, optimizer, criterion, device)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            acc = evaluate_centralized(model, testloader, device)
            print(f"Epoch {epoch:3d}/{args.epochs} | loss {loss:.4f} | test acc {acc*100:.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\n[DONE] Đã lưu mô hình centralized tại: {args.output}")


if __name__ == '__main__':
    main()
