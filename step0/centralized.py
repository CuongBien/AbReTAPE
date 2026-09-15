import os
import sys
import time
import json
import csv
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
from plot import plot_history


def parse_args():
    parser = argparse.ArgumentParser(description="Centralized Training on CIFAR-10 (ResNet-18)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (mặc định: 0.1)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá test (mặc định: 5)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (mặc định: 2)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(SCRIPT_DIR, "data"), help="Thư mục chứa CIFAR-10")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "b0_centralized.pt"), help="File lưu checkpoint tham chiếu")
    parser.add_argument("--save-best", type=str, default=os.path.join(SCRIPT_DIR, "best_b0_centralized.pt"), help="File lưu best checkpoint")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(SCRIPT_DIR, "last_checkpoint_centralized.pt"), help="File checkpoint resume")
    parser.add_argument("--history-file", type=str, default=os.path.join(SCRIPT_DIR, "centralized_history.json"), help="File lưu lịch sử JSON")
    parser.add_argument("--plot-file", type=str, default=os.path.join(SCRIPT_DIR, "centralized_curves.png"), help="File lưu ảnh đồ thị")
    parser.add_argument("--no-plot", action="store_true", help="Không tự động vẽ đồ thị")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint nếu có")
    return parser.parse_args()


def train_epoch_centralized(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_centralized(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    val_loss = (total_loss / total) if criterion is not None else 0.0
    return val_loss, correct / total


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("==================================================", flush=True)
    print("Chế độ           : Centralized Training (Tham chiếu)", flush=True)
    print(f"Thiết bị         : {device}", flush=True)
    if device == 'cuda':
        print(f"GPU Name         : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Epochs           : {args.epochs}", flush=True)
    print(f"Batch size       : {args.batch_size}", flush=True)
    print(f"Learning rate    : {args.lr}", flush=True)
    print(f"Eval freq        : every {args.eval_freq} epochs", flush=True)
    print(f"Output           : {args.output}", flush=True)
    print(f"Best model       : {args.save_best}", flush=True)
    print(f"Checkpoint       : {args.checkpoint} (resume={args.resume})", flush=True)
    print("==================================================", flush=True)

    torch.manual_seed(0)
    trainloader, testloader = get_cifar10(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = resnet18_cifar().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_acc = 0.0
    history = []

    if args.resume and os.path.isfile(args.checkpoint):
        print(f"[RESUME] Đang nạp checkpoint từ: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        history = ckpt.get('history', [])
        if not history and os.path.isfile(args.history_file):
            try:
                with open(args.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception:
                history = []
        print(f"[RESUME] Tiếp tục từ epoch {start_epoch} (Best Acc: {best_acc*100:.2f}%)", flush=True)

    print("Bắt đầu huấn luyện Centralized...\n", flush=True)
    total_start = time.time()
    csv_file = os.path.splitext(args.history_file)[0] + ".csv"

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch_centralized(model, trainloader, optimizer, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        epoch_time = time.time() - t0

        test_loss = None
        test_acc = None
        eval_time = 0.0
        best_tag = ""

        is_eval = (epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs)
        if is_eval:
            t_eval_0 = time.time()
            test_loss, test_acc = evaluate_centralized(model, testloader, device, criterion=criterion)
            eval_time = time.time() - t_eval_0
            if test_acc > best_acc:
                best_acc = test_acc
                os.makedirs(os.path.dirname(os.path.abspath(args.save_best)), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'history': history,
                }, args.save_best)
                best_tag = " -> [BEST SAVED]"

            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | LR: {current_lr:.4f} | Train: {epoch_time:.1f}s | Eval: {eval_time:.1f}s{best_tag}", flush=True)
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | LR: {current_lr:.4f} | Train: {epoch_time:.1f}s", flush=True)

        epoch_entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'test_loss': float(test_loss) if test_loss is not None else None,
            'test_acc': float(test_acc) if test_acc is not None else None,
            'lr': float(current_lr),
            'epoch_time': float(epoch_time + eval_time),
        }
        history.append(epoch_entry)

        os.makedirs(os.path.dirname(os.path.abspath(args.history_file)), exist_ok=True)
        with open(args.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr", "epoch_time"])
                writer.writeheader()
                writer.writerows(history)
        except Exception:
            pass

        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'history': history,
        }, args.checkpoint)

        if not args.no_plot and (is_eval or epoch == args.epochs):
            try:
                plot_history(history, save_path=args.plot_file, show=False)
            except Exception as e:
                print(f"[WARN] Lỗi khi vẽ đồ thị: {e}", flush=True)

    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Accuracy: {best_acc*100:.2f}%", flush=True)

    if not args.no_plot:
        try:
            plot_history(history, save_path=args.plot_file, show=False)
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({'model': model.state_dict(), 'best_acc': best_acc, 'history': history}, args.output)
    print(f"[DONE] Đã lưu mô hình centralized tại: {args.output}", flush=True)
    print(f"[DONE] Đã lưu lịch sử huấn luyện tại: {args.history_file} & {csv_file}", flush=True)
    if not args.no_plot:
        print(f"[DONE] Đã lưu đồ thị tại: {args.plot_file}\n", flush=True)


if __name__ == '__main__':
    main()
