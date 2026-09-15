# Bước 0 — chạy vanilla Split Learning trên CIFAR-10
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

# Thêm thư mục chứa file vào sys.path để import an toàn khi chạy từ bất kỳ đâu
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn
from model import ClientModel, ServerModel
from data import get_cifar10
from train import train_epoch, evaluate
from plot import plot_history


def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla Split Learning on CIFAR-10 (ResNet-18)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Test evaluation frequency (epochs, default: 5)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(SCRIPT_DIR, "data"), help="CIFAR-10 data directory")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "b0_vanilla.pt"), help="Final reference checkpoint path")
    parser.add_argument("--save-best", type=str, default=os.path.join(SCRIPT_DIR, "best_b0_vanilla.pt"), help="Best accuracy model checkpoint path")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(SCRIPT_DIR, "last_checkpoint.pt"), help="Periodic checkpoint path for resume")
    parser.add_argument("--history-file", type=str, default=os.path.join(SCRIPT_DIR, "history.json"), help="Training history JSON path")
    parser.add_argument("--plot-file", type=str, default=os.path.join(SCRIPT_DIR, "training_curves.png"), help="Output plot image path")
    parser.add_argument("--no-plot", action="store_true", help="Disable automatic plot generation")
    parser.add_argument("--resume", action="store_true", help="Resume training from last_checkpoint.pt if available")
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
    print(f"Final output     : {args.output}", flush=True)
    print(f"Best model path  : {args.save_best}", flush=True)
    print(f"Resume checkpoint: {args.checkpoint} (resume={args.resume})", flush=True)
    print(f"History file     : {args.history_file}", flush=True)
    print(f"Plot file        : {args.plot_file} (no_plot={args.no_plot})", flush=True)
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

    start_epoch = 1
    best_acc = 0.0
    history = []

    # Khôi phục nếu bật cờ --resume và file checkpoint tồn tại
    if args.resume and os.path.isfile(args.checkpoint):
        print(f"[RESUME] Loading checkpoint from: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device)
        client.load_state_dict(ckpt['client'])
        server.load_state_dict(ckpt['server'])
        opt_c.load_state_dict(ckpt['opt_c'])
        opt_s.load_state_dict(ckpt['opt_s'])
        sched_c.load_state_dict(ckpt['sched_c'])
        sched_s.load_state_dict(ckpt['sched_s'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        history = ckpt.get('history', [])
        if not history and os.path.isfile(args.history_file):
            try:
                with open(args.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception:
                history = []
        print(f"[RESUME] Resuming from epoch {start_epoch} (Best Acc so far: {best_acc*100:.2f}%, History: {len(history)} epochs)", flush=True)

    print("Starting Vanilla Split Learning training...\n", flush=True)
    total_start = time.time()
    csv_file = os.path.splitext(args.history_file)[0] + ".csv"

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(client, server, trainloader, opt_c, opt_s, criterion, device)
        current_lr = opt_c.param_groups[0]['lr']
        sched_c.step()
        sched_s.step()
        epoch_time = time.time() - t0

        test_loss = None
        test_acc = None
        eval_time = 0.0
        best_tag = ""

        # Đánh giá độ chính xác định kỳ hoặc ở epoch đầu/cuối
        is_eval_epoch = (epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs)
        if is_eval_epoch:
            t_eval_0 = time.time()
            test_loss, test_acc = evaluate(client, server, testloader, device, criterion=criterion)
            eval_time = time.time() - t_eval_0

            # Lưu best checkpoint nếu đạt accuracy cao nhất
            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
                os.makedirs(os.path.dirname(os.path.abspath(args.save_best)), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'client': client.state_dict(),
                    'server': server.state_dict(),
                    'opt_c': opt_c.state_dict(),
                    'opt_s': opt_s.state_dict(),
                    'sched_c': sched_c.state_dict(),
                    'sched_s': sched_s.state_dict(),
                    'best_acc': best_acc,
                    'history': history + [{
                        'epoch': epoch,
                        'train_loss': float(train_loss),
                        'train_acc': float(train_acc),
                        'test_loss': float(test_loss),
                        'test_acc': float(test_acc),
                        'lr': float(current_lr),
                        'epoch_time': float(epoch_time + eval_time),
                    }],
                }, args.save_best)
                best_tag = " -> [BEST SAVED]"

            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | LR: {current_lr:.4f} | Train: {epoch_time:.1f}s | Eval: {eval_time:.1f}s{best_tag}", flush=True)
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | LR: {current_lr:.4f} | Train: {epoch_time:.1f}s", flush=True)

        # Ghi nhận vào history
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

        # Lưu history ra file JSON
        os.makedirs(os.path.dirname(os.path.abspath(args.history_file)), exist_ok=True)
        with open(args.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        # Lưu history ra file CSV
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr", "epoch_time"])
                writer.writeheader()
                writer.writerows(history)
        except Exception:
            pass

        # Tự động lưu checkpoint ngắt quãng sau mỗi epoch để có thể resume bất kỳ lúc nào
        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'client': client.state_dict(),
            'server': server.state_dict(),
            'opt_c': opt_c.state_dict(),
            'opt_s': opt_s.state_dict(),
            'sched_c': sched_c.state_dict(),
            'sched_s': sched_s.state_dict(),
            'best_acc': best_acc,
            'history': history,
        }, args.checkpoint)

        # Cập nhật đồ thị trực quan định kỳ sau mỗi lần eval hoặc ở epoch cuối
        if not args.no_plot and (is_eval_epoch or epoch == args.epochs):
            try:
                plot_history(history, save_path=args.plot_file, show=False)
            except Exception as e:
                print(f"[WARN] Lỗi khi vẽ đồ thị: {e}", flush=True)

    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Accuracy: {best_acc*100:.2f}%", flush=True)

    # Đảm bảo đồ thị được vẽ và lưu lần cuối
    if not args.no_plot:
        try:
            plot_history(history, save_path=args.plot_file, show=False)
        except Exception as e:
            pass

    # Lưu trọng số tham chiếu cuối cùng (kèm theo best_acc và history)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        'client': client.state_dict(),
        'server': server.state_dict(),
        'best_acc': best_acc,
        'history': history,
    }, args.output)
    print(f"[DONE] Saved final reference weights to: {args.output}", flush=True)
    print(f"[DONE] Saved training history to: {args.history_file} & {csv_file}", flush=True)
    if not args.no_plot:
        print(f"[DONE] Saved visualization plot to: {args.plot_file}\n", flush=True)



if __name__ == '__main__':
    main()
