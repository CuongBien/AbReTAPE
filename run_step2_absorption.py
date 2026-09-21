#!/usr/bin/env python3
# Bước 2: Thí nghiệm Hấp thụ (Absorption Experiment) — Chứng minh tính thất bại của phép mã hóa khả nghịch
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

from src.models import ClientModel, ServerModel
from src.data import get_cifar10, CIFAR10_MEAN, CIFAR10_STD
from src.defenses import random_perm, ChannelPermute, Adapter
from src.training import train_sl_epoch, evaluate_sl
from src.attacks import (
    Decoder,
    recover_perm,
    recover_perm_from_adapter,
    recover_perm_covariance,
    match_accuracy,
    evaluate_inversion
)
from src.utils import plot_training_curves, plot_permutation_matrix


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
    parser = argparse.ArgumentParser(description="Bước 2: Thí nghiệm Hấp thụ ánh xạ ngược (Absorption Experiment)")
    parser.add_argument("--mode", choices=["adapter", "train_sl"], default="adapter",
                        help="'adapter': Huấn luyện Cut-Layer Adapter chứng minh hấp thụ (khuyên dùng); 'train_sl': Huấn luyện SL với Permutation")
    parser.add_argument("--epochs", type=int, default=15, help="Số epochs huấn luyện (mặc định: 15 cho adapter, 20 cho train_sl)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (mặc định: 0.01)")
    parser.add_argument("--perm-seed", type=int, default=42, help="Seed sinh hoán vị ngẫu nhiên (mặc định: 42)")
    parser.add_argument("--ref-ckpt", type=str, default=None, help="File checkpoint Step 0 (b0_vanilla.pt)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục dữ liệu")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step2"), help="Thư mục outputs")
    return parser.parse_args()


def run_adapter_mode(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = find_step0_checkpoint(args.ref_ckpt)
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Không tìm thấy checkpoint Step 0 tại: {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print("BƯỚC 2: CHỨNG MINH HẤP THỤ QUA CUT-LAYER ADAPTER (CONV 1x1)")
    print(f"Thiết bị: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Nạp trọng số hội tụ từ: {ckpt_path}")
    print("=" * 70)

    # 1. Nạp và đóng băng Client & Server
    client = ClientModel().to(device)
    server = ServerModel().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    client.load_state_dict(ckpt["client"] if "client" in ckpt else ckpt)
    server.load_state_dict(ckpt["server"] if "server" in ckpt else ckpt)

    client.eval()
    server.eval()
    for p in client.parameters():
        p.requires_grad = False
    for p in server.parameters():
        p.requires_grad = False

    # 2. Tạo hoán vị bí mật pi và Adapter 1x1 conv
    perm = random_perm(64, seed=args.perm_seed).to(device)
    permute = ChannelPermute(perm).to(device)
    adapter = Adapter(channels=64).to(device)

    opt_a = torch.optim.Adam(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_a = CosineAnnealingLR(opt_a, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size)

    # Thử nghiệm thống kê nhanh Covariance Attack (0 epochs)
    cov_perm_hat, cov_acc = recover_perm_covariance(client, trainloader, perm, device, num_batches=30)
    print(f"\n[PHƯƠNG PHÁP 1: COVARIANCE PROFILE (0 EPOCHS)]")
    print(f"Độ khớp khôi phục hoán vị tức thì: {cov_acc*100:.2f}%\n")

    print(f"[PHƯƠNG PHÁP 2: HUẤN LUYỆN CUT-LAYER ADAPTER (HẤP THỤ ÁNH XẠ NGƯỢC)]")
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt_a.zero_grad()
            with torch.no_grad():
                z_perm = permute(client(x))
            z_hat = adapter(z_perm)
            logits = server(z_hat)
            loss = criterion(logits, y)
            loss.backward()
            opt_a.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        sched_a.step()
        epoch_time = time.time() - t0

        # Kiểm tra độ khớp
        A = adapter.get_matrix()
        hat_pi = recover_perm_from_adapter(A)
        match_acc = match_accuracy(perm, hat_pi)

        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}% | "
              f"Độ khớp Hoán vị khôi phục: {match_acc*100:.1f}% | Time: {epoch_time:.1f}s", flush=True)

    # Lưu kết quả
    heatmap_file = os.path.join(args.output_dir, "adapter_heatmap.png")
    plot_permutation_matrix(adapter.get_matrix(), save_path=heatmap_file, title="Ma trận Trọng số Adapter A ≈ P_pi^T (Hấp thụ hoàn toàn)")

    out_file = os.path.join(args.output_dir, "b2_adapter.pt")
    torch.save({
        "adapter": adapter.state_dict(),
        "A": adapter.get_matrix().cpu(),
        "perm": perm.cpu(),
        "perm_hat": hat_pi,
        "match_acc": match_acc,
        "cov_match_acc": cov_acc
    }, out_file)
    print(f"\n[DONE] Hoàn thành thí nghiệm Hấp thụ Adapter! Độ khớp hoán vị khôi phục: {match_acc*100:.1f}%")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_adapter_mode(args, device)


if __name__ == "__main__":
    main()
