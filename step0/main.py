# Bước 0 — chạy vanilla Split Learning trên CIFAR-10
import os
import sys
import time
import argparse

# Thiết lập UTF-8 encoding cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục chứa file vào sys.path để import an toàn khi chạy từ bất kỳ đâu
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn
from model import ClientModel, ServerModel
from data import get_cifar10
from train import train_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla Split Learning on CIFAR-10 (ResNet-18)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Test evaluation frequency (epochs, default: 5)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(SCRIPT_DIR, "data"), help="CIFAR-10 data directory")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "b0_vanilla.pt"), help="Checkpoint save path")
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("==================================================", flush=True)
    print(f"Device           : {device}", flush=True)
    if device == 'cuda':
        print(f"GPU Name         : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Epochs           : {args.epochs}", flush=True)
    print(f"Batch size       : {args.batch_size}", flush=True)
    print(f"Learning rate    : {args.lr}", flush=True)
    print(f"Eval frequency   : every {args.eval_freq} epochs", flush=True)
    print(f"Checkpoint output: {args.output}", flush=True)
    print("==================================================", flush=True)

    torch.manual_seed(0)
    trainloader, testloader = get_cifar10(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    client = ClientModel().to(device)
    server = ServerModel().to(device)

    # Cùng bộ siêu tham số cho cả hai phía (SL chuẩn)
    opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    sched_c = torch.optim.lr_scheduler.MultiStepLR(opt_c, milestones=[50, 75], gamma=0.1)
    sched_s = torch.optim.lr_scheduler.MultiStepLR(opt_s, milestones=[50, 75], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    print("Starting Vanilla Split Learning training...\n", flush=True)
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(client, server, trainloader, opt_c, opt_s, criterion, device)
        sched_c.step()
        sched_s.step()
        epoch_time = time.time() - t0

        # Luôn in từng epoch để người dùng thấy rõ tiến trình
        if epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs:
            t_eval_0 = time.time()
            acc = evaluate(client, server, testloader, device)
            eval_time = time.time() - t_eval_0
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {loss:.4f} | Test Acc: {acc*100:.2f}% | Train Time: {epoch_time:.1f}s | Eval Time: {eval_time:.1f}s", flush=True)
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {loss:.4f} | Train Time: {epoch_time:.1f}s", flush=True)

    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time/60:.2f} minutes.", flush=True)

    # Lưu trọng số tham chiếu (dùng cho Bước 2 — thí nghiệm hấp thụ)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({'client': client.state_dict(), 'server': server.state_dict()}, args.output)
    print(f"[DONE] Saved reference weights to: {args.output}\n", flush=True)


if __name__ == '__main__':
    main()
