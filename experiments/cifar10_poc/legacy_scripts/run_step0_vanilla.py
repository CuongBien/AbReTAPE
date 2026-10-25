#!/usr/bin/env python3
# Bước 0: Huấn luyện Vanilla Split Learning (B0) và Centralized Training đối chứng
import os
import sys
import time
import json
import csv
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hỗ trợ UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel, ServerModel, resnet18_cifar
from src.data import get_cifar10
from src.training import train_sl_epoch, evaluate_sl, train_centralized_epoch, evaluate_centralized, EarlyStopping
from src.utils import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 0: Huấn luyện Vanilla Split Learning hoặc Centralized đối chứng")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (mặc định: 0.1)")
    parser.add_argument("--eval-freq", type=int, default=1, help="Tần suất đánh giá test set (mặc định: 1)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2,
                        help="Số luồng nạp dữ liệu (mặc định: 0 trên Windows để tránh crash IPC, 2 trên Linux)")
    parser.add_argument("--patience", type=int, default=15, help="Số epochs chờ Early Stopping (mặc định: 15, 0 để tắt)")
    parser.add_argument("--centralized", action="store_true", help="Chạy chế độ Centralized Training đối chứng thay vì SL")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục dữ liệu")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step0"), help="Thư mục lưu outputs")
    parser.add_argument("--resume", action="store_true", help="Khôi phục huấn luyện từ checkpoint gần nhất")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn lưu/nạp checkpoint dở dang")
    parser.add_argument("--save-best", type=str, default=None, help="Đường dẫn lưu best model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu model checkpoint cuối cùng")
    parser.add_argument("--history-file", type=str, default=None, help="Đường dẫn lưu file lịch sử JSON")
    parser.add_argument("--plot-file", type=str, default=None, help="Đường dẫn lưu file đồ thị PNG")
    return parser.parse_args()


def run_centralized(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = args.save_best or os.path.join(args.output_dir, "best_b0_centralized.pt")
    last_path = args.checkpoint or os.path.join(args.output_dir, "last_checkpoint_centralized.pt")
    final_path = args.output or os.path.join(args.output_dir, "b0_centralized.pt")
    history_json = args.history_file or os.path.join(args.output_dir, "centralized_history.json")
    history_csv = os.path.join(args.output_dir, "centralized_history.csv")
    plot_file = args.plot_file or os.path.join(args.output_dir, "centralized_curves.png")

    print("=" * 70)
    print(f"BƯỚC 0: CENTRALIZED TRAINING ĐỐI CHỨNG (CIFAR-10, ResNet-18)")
    print(f"Thiết bị: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("=" * 70)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    model = resnet18_cifar().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_acc = 0.0
    history = []
    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    if args.resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        history = ckpt.get("history", [])
        print(f"[RESUME] Đã khôi phục từ epoch {start_epoch-1} với Best Acc: {best_acc*100:.2f}%")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_centralized_epoch(model, trainloader, optimizer, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        test_loss, test_acc = None, None
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            test_loss, test_acc = evaluate_centralized(model, testloader, device, criterion)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({"model": model.state_dict(), "best_acc": best_acc, "epoch": epoch}, best_path)
            if early_stopping.step(test_acc, epoch=epoch):
                print(f"\n[EARLY STOPPING] Dừng sớm tại epoch {epoch} do test acc không cải thiện sau {args.patience} lần đánh giá! Best Acc: {best_acc*100:.2f}% (Epoch {early_stopping.best_epoch})")
                break

        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss) if test_loss is not None else None,
            "test_acc": float(test_acc) if test_acc is not None else None,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time": float(epoch_time),
        }
        history.append(entry)

        status = f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
        if test_acc is not None:
            status += f" | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% (Best: {best_acc*100:.2f}%)"
        print(status + f" | Time: {epoch_time:.1f}s", flush=True)

        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "history": history
        }, last_path)

    torch.save({"model": model.state_dict(), "best_acc": best_acc}, final_path)
    plot_training_curves(history, plot_file)
    print(f"\n[DONE] Hoàn thành Centralized Training! Best Acc: {best_acc*100:.2f}%")


def run_split_learning(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = args.save_best or os.path.join(args.output_dir, "best_b0_vanilla.pt")
    last_path = args.checkpoint or os.path.join(args.output_dir, "last_checkpoint.pt")
    final_path = args.output or os.path.join(args.output_dir, "b0_vanilla.pt")
    history_json = args.history_file or os.path.join(args.output_dir, "history.json")
    plot_file = args.plot_file or os.path.join(args.output_dir, "training_curves.png")

    print("=" * 70)
    print(f"BƯỚC 0: VANILLA SPLIT LEARNING (CIFAR-10, ResNet-18)")
    print(f"Thiết bị: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Thư mục Output: {args.output_dir}")
    print("=" * 70)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    client = ClientModel().to(device)
    server = ServerModel().to(device)

    opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    sched_c = CosineAnnealingLR(opt_c, T_max=args.epochs)
    sched_s = CosineAnnealingLR(opt_s, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_acc = 0.0
    history = []
    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    if args.resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        client.load_state_dict(ckpt["client"])
        server.load_state_dict(ckpt["server"])
        opt_c.load_state_dict(ckpt["opt_c"])
        opt_s.load_state_dict(ckpt["opt_s"])
        sched_c.load_state_dict(ckpt["sched_c"])
        sched_s.load_state_dict(ckpt["sched_s"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        history = ckpt.get("history", [])
        print(f"[RESUME] Đã khôi phục từ epoch {start_epoch-1} với Best Acc: {best_acc*100:.2f}%")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_sl_epoch(client, server, trainloader, opt_c, opt_s, criterion, device)
        sched_c.step()
        sched_s.step()
        epoch_time = time.time() - t0

        test_loss, test_acc = None, None
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            test_loss, test_acc = evaluate_sl(client, server, testloader, device, criterion=criterion)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({"client": client.state_dict(), "server": server.state_dict(), "best_acc": best_acc, "epoch": epoch}, best_path)
            if early_stopping.step(test_acc, epoch=epoch):
                print(f"\n[EARLY STOPPING] Dừng sớm tại epoch {epoch} do test acc không cải thiện sau {args.patience} lần đánh giá! Best Acc: {best_acc*100:.2f}% (Epoch {early_stopping.best_epoch})")
                break

        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss) if test_loss is not None else None,
            "test_acc": float(test_acc) if test_acc is not None else None,
            "lr": float(opt_s.param_groups[0]["lr"]),
            "epoch_time": float(epoch_time),
        }
        history.append(entry)

        status = f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
        if test_acc is not None:
            status += f" | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% (Best: {best_acc*100:.2f}%)"
        print(status + f" | Time: {epoch_time:.1f}s", flush=True)

        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        torch.save({
            "epoch": epoch,
            "client": client.state_dict(),
            "server": server.state_dict(),
            "opt_c": opt_c.state_dict(),
            "opt_s": opt_s.state_dict(),
            "sched_c": sched_c.state_dict(),
            "sched_s": sched_s.state_dict(),
            "best_acc": best_acc,
            "history": history
        }, last_path)

    torch.save({"client": client.state_dict(), "server": server.state_dict(), "best_acc": best_acc}, final_path)
    plot_training_curves(history, plot_file)
    print(f"\n[DONE] Hoàn thành Vanilla Split Learning! Best Acc: {best_acc*100:.2f}%")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.centralized:
        run_centralized(args, device)
    else:
        run_split_learning(args, device)


if __name__ == "__main__":
    main()
