#!/usr/bin/env python3
# Bước 1: Huấn luyện bộ giải mã Feature Inversion Attack (Passive Reconstruction Attack)
import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel
from src.data import get_cifar10, CIFAR10_MEAN, CIFAR10_STD
from src.attacks import Decoder, train_inversion_epoch, evaluate_inversion
from src.training import EarlyStopping
from src.metrics import get_lpips_fn
from src.utils import plot_attack_curves, save_reconstruction_grid


def find_step0_checkpoint(user_path=None):
    if user_path and os.path.isfile(user_path):
        return user_path
    candidates = [
        os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step0", "best_b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step0", "b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "best_b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "b0_vanilla.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 1: Feature Inversion Attack trên Vanilla Split Learning")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện Decoder (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate cho Decoder (mặc định: 1e-3)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá test set (mặc định: 5)")
    parser.add_argument("--patience", type=int, default=10, help="Số lần đánh giá chờ Early Stopping (mặc định: 10, 0 để tắt)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2,
                        help="Số luồng nạp dữ liệu (mặc định: 0 trên Windows để tránh crash IPC, 2 trên Linux)")
    parser.add_argument("--client-ckpt", type=str, default=None, help="Đường dẫn file checkpoint Step 0")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục dữ liệu")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step1"), help="Thư mục lưu outputs")
    parser.add_argument("--resume", action="store_true", help="Khôi phục huấn luyện từ checkpoint gần nhất")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn lưu/nạp checkpoint dở dang")
    parser.add_argument("--save-best", type=str, default=None, help="Đường dẫn lưu best decoder checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu decoder checkpoint cuối cùng")
    parser.add_argument("--history-file", type=str, default=None, help="Đường dẫn lưu file lịch sử JSON")
    parser.add_argument("--plot-file", type=str, default=None, help="Đường dẫn lưu file đồ thị PNG")
    parser.add_argument("--grid-file", type=str, default=None, help="Đường dẫn lưu ảnh lưới tái tạo PNG")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = find_step0_checkpoint(args.client_ckpt)
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Không tìm thấy checkpoint Step 0 tại: {ckpt_path}")
        print("Hãy chạy 'python run_step0_vanilla.py' trước để huấn luyện mô hình nền tảng.")
        sys.exit(1)

    print("=" * 70)
    print("BƯỚC 1: FEATURE INVERSION ATTACK (THREAT MODEL: HBC PASSIVE RECONSTRUCTION)")
    print(f"Thiết bị: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Nạp Client weights từ: {ckpt_path}")
    print(f"Thư mục Output: {args.output_dir}")
    print("=" * 70)

    # 1. Nạp Client model và đóng băng
    client = ClientModel().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    client_weights = ckpt["client"] if "client" in ckpt else ckpt
    client.load_state_dict(client_weights)
    client.eval()
    for p in client.parameters():
        p.requires_grad = False
    print("[INFO] Đã đóng băng hoàn toàn ClientModel.")

    # 2. Khởi tạo Decoder và Optimizer
    decoder = Decoder(in_channels=64, out_channels=3, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    lpips_fn = get_lpips_fn(device=device)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    best_psnr = 0.0
    history = []
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    best_path = args.save_best or os.path.join(args.output_dir, "best_b1_decoder.pt")
    last_path = args.checkpoint or os.path.join(args.output_dir, "last_checkpoint_b1.pt")
    final_path = args.output or os.path.join(args.output_dir, "b1_decoder.pt")
    history_json = args.history_file or os.path.join(args.output_dir, "step1_history.json")
    plot_file = args.plot_file or os.path.join(args.output_dir, "attack_curves.png")
    grid_file = args.grid_file or os.path.join(args.output_dir, "reconstruction_grid.png")

    start_epoch = 1
    if args.resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        history = ckpt.get("history", [])
        print(f"[RESUME] Đã khôi phục huấn luyện Decoder từ epoch {start_epoch-1} với Best PSNR: {best_psnr:.2f} dB")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_mse = train_inversion_epoch(client, decoder, trainloader, optimizer, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        test_mse, test_psnr, test_ssim, test_lpips = None, None, None, None
        is_eval = (epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs)
        if is_eval:
            test_mse, test_psnr, test_ssim, test_lpips = evaluate_inversion(
                client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD,
                criterion=criterion, lpips_fn=lpips_fn
            )
            if test_psnr is not None and test_psnr > best_psnr:
                best_psnr = test_psnr
                torch.save({"decoder": decoder.state_dict(), "best_psnr": best_psnr, "epoch": epoch}, best_path)
            if test_psnr is not None and early_stopping.step(test_psnr, epoch=epoch):
                print(f"\n[EARLY STOPPING] Dừng sớm Decoder tại epoch {epoch} do PSNR không cải thiện sau {args.patience} lần đánh giá! Best PSNR: {best_psnr:.2f} dB")
                break

        entry = {
            "epoch": epoch,
            "train_mse": float(train_mse),
            "test_mse": float(test_mse) if test_mse is not None else None,
            "test_psnr": float(test_psnr) if test_psnr is not None else None,
            "test_ssim": float(test_ssim) if test_ssim is not None else None,
            "test_lpips": float(test_lpips) if test_lpips is not None else None,
            "epoch_time": float(epoch_time),
        }
        history.append(entry)

        status = f"Epoch {epoch:2d}/{args.epochs} | Train MSE: {train_mse:.4f}"
        if test_psnr is not None:
            status += f" | Test MSE: {test_mse:.4f} | PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f}"
            if test_lpips is not None:
                status += f" | LPIPS: {test_lpips:.4f}"
        print(status + f" | Time: {epoch_time:.1f}s", flush=True)

        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        torch.save({
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "history": history
        }, last_path)

    torch.save({"decoder": decoder.state_dict(), "best_psnr": best_psnr}, final_path)
    plot_attack_curves(history, plot_file)
    save_reconstruction_grid(client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=grid_file, num_images=8)
    print(f"\n[DONE] Hoàn thành Feature Inversion Attack! Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
