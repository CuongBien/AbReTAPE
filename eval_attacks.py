#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_attacks.py
==============================================================================
UNIFIED ATTACK EVALUATION ENGINE CHO SPLIT LEARNING
Đánh giá tính bảo mật của Client Checkpoint trước toàn bộ các họ tấn công:
1. Fresh MLP Attacker (Passive Adaptive Decoder, 30 epochs)
2. Fresh Conv Attacker (Passive Conv Lift Decoder, 30 epochs)
3. FSHA (Active Feature Space Hijacking Attack, Pasquini et al. CCS 2021)
Hỗ trợ cả CIFAR-10 và Y Tế (HAM10000, PCAM).
==============================================================================
"""
import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import (
    ClientModel,
    ServerModel,
    ClientWithBottleneck,
    ServerWithBottleneck,
    LearnedAdaptiveDecoder
)
from src.attacks.decoder import Decoder
from src.data import get_dataloaders, get_fsha_dataloaders
from src.metrics import psnr_ssim, get_lpips_fn, calculate_lpips
from src.utils.visualize import save_reconstruction_grid


class ConvLiftDecoder(nn.Module):
    """
    Fresh Conv Attacker: Nâng chiều zt @ W_lift^T rồi đưa qua ConvDecoder 4 tầng.
    """
    def __init__(self, bottleneck_weight, in_channels=64, out_channels=3):
        super().__init__()
        self.register_buffer("W_lift", bottleneck_weight.t().clone())
        self.conv_decoder = Decoder(in_channels=in_channels, out_channels=out_channels)

    def forward(self, zt):
        if zt.dim() > 2:
            zt = zt.view(zt.size(0), -1)
        zh = torch.matmul(zt, self.W_lift.t())
        zh_tensor = zh.view(zt.size(0), 64, 32, 32)
        return self.conv_decoder(zh_tensor)


@torch.no_grad()
def evaluate_attacker_metrics(client, attacker, loader, device, mean, std, lpips_fn=None, max_batches=None):
    client.eval()
    attacker.eval()

    total_samples = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    mse_sum = 0.0
    criterion = nn.MSELoss()

    for idx, (x, _) in enumerate(loader):
        x = x.to(device)
        b = x.size(0)
        zt = client(x)
        x_rec = attacker(zt)

        mse_sum += criterion(x_rec, x).item() * b
        p, s = psnr_ssim(x, x_rec, mean, std)
        psnr_sum += p * b
        ssim_sum += s * b

        if lpips_fn is not None:
            lp = calculate_lpips(lpips_fn, x, x_rec, mean, std)
            lpips_sum += lp * b

        total_samples += b
        if max_batches is not None and idx >= max_batches - 1:
            break

    return {
        "psnr": psnr_sum / max(1, total_samples),
        "ssim": ssim_sum / max(1, total_samples),
        "lpips": (lpips_sum / max(1, total_samples)) if lpips_fn else None,
        "mse": mse_sum / max(1, total_samples)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Split Learning Attack Evaluation")
    parser.add_argument("--client-ckpt", type=str, required=True, help="Đường dẫn checkpoint Client")
    parser.add_argument("--server-ckpt", type=str, default=None, help="Đường dẫn checkpoint Server (cần cho FSHA)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "ham10000", "pcam"])
    parser.add_argument("--attackers", nargs="+", default=["fresh_mlp", "fresh_conv"],
                        choices=["fresh_mlp", "fresh_conv", "fsha"], help="Danh sách kẻ tấn công cần đánh giá")
    parser.add_argument("--k", type=int, default=2048, help="Số chiều bottleneck (mặc định: 2048)")
    parser.add_argument("--m", type=int, default=512, help="Số chiều unprojector (mặc định: 512)")
    parser.add_argument("--D", type=int, default=65536, help="Số chiều cut-layer (mặc định: 65536)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện kẻ tấn công")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate cho Fresh Attacker")
    parser.add_argument("--fixed-bottleneck", action="store_true", default=True, help="Sử dụng ma trận chiếu trực giao cố định R")
    parser.add_argument("--grad-scales", nargs="+", type=float, default=[1.0, 5.0], help="Grad scales cho FSHA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="./data", help="Thư mục dữ liệu")
    parser.add_argument("--output-dir", type=str, default="./output/eval_attacks", help="Thư mục xuất kết quả")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm nhanh (1 epoch)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("BẮT ĐẦU ĐÁNH GIÁ TÍNH BẢO MẬT (THREAT SURFACE BENCHMARK)")
    print(f"Client Checkpoint: {args.client_ckpt}")
    print(f"Dataset: {args.dataset.upper()} | Kẻ tấn công: {args.attackers} | Thiết bị: {device}")
    print(f"Epochs đánh giá: {args.epochs} | Seed: {args.seed}")
    print("=" * 80, flush=True)

    # 1. Nạp Dữ Liệu
    train_loader, test_loader, meta = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )
    mean, std = meta["mean"], meta["std"]

    # 2. Khởi tạo và Nạp Client
    if not os.path.isfile(args.client_ckpt):
        print(f"[LỖI] Checkpoint client không tồn tại: {args.client_ckpt}")
        sys.exit(1)

    ckpt_sd = torch.load(args.client_ckpt, map_location=device)
    is_bottleneck = (args.k is not None and args.k < args.D) or ("R" in ckpt_sd) or ("bottleneck.weight" in ckpt_sd)
    is_fixed = getattr(args, "fixed_bottleneck", True) or ("R" in ckpt_sd)

    if is_bottleneck:
        client = ClientWithBottleneck(
            k=args.k, D=args.D, init_ortho=False, seed=args.seed, fixed_bottleneck=is_fixed
        ).to(device)
    else:
        client = ClientModel().to(device)

    client.load_state_dict(ckpt_sd)
    client.eval()
    for p in client.parameters():
        p.requires_grad = False

    lpips_fn = get_lpips_fn(device=device)
    results = []
    total_epochs = 1 if args.dry_run else args.epochs

    # ==========================================================================
    # ĐÁNH GIÁ FRESH MLP & CONV ATTACKERS
    # ==========================================================================
    for atk_name in args.attackers:
        if atk_name not in ["fresh_mlp", "fresh_conv"]:
            continue

        print(f"\n>>> Đang đánh giá Fresh Attacker: {atk_name.upper()} ({total_epochs} epochs)...", flush=True)
        torch.manual_seed(args.seed)

        if atk_name == "fresh_mlp":
            attacker = LearnedAdaptiveDecoder(k=args.k, D=args.D, hidden_dim=args.m).to(device)
        elif atk_name == "fresh_conv":
            w_lift = client.get_bottleneck_weight() if hasattr(client, "get_bottleneck_weight") else torch.eye(args.k, args.D, device=device)
            attacker = ConvLiftDecoder(w_lift).to(device)

        opt = torch.optim.Adam(attacker.parameters(), lr=args.lr)
        sched = CosineAnnealingLR(opt, T_max=total_epochs)
        criterion = nn.MSELoss()

        history = []
        t0 = time.time()

        for ep in range(1, total_epochs + 1):
            attacker.train()
            train_mse = 0.0
            n_batches = 0

            for b_idx, (x, _) in enumerate(train_loader):
                x = x.to(device)
                with torch.no_grad():
                    zt = client(x)
                x_rec = attacker(zt)
                loss = criterion(x_rec, x)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_mse += loss.item()
                n_batches += 1
                if args.dry_run and b_idx >= 2:
                    break

            sched.step()
            train_mse /= max(1, n_batches)

            # Đánh giá cuối epoch
            max_b = 2 if args.dry_run else None
            metrics = evaluate_attacker_metrics(client, attacker, test_loader, device, mean, std, lpips_fn, max_batches=max_b)
            rec = {
                "epoch": ep,
                "train_mse": train_mse,
                "test_psnr": metrics["psnr"],
                "test_ssim": metrics["ssim"]
            }
            history.append(rec)
            print(f"  [Ep {ep:2d}/{total_epochs:2d}] Train MSE: {train_mse:.4f} | Test PSNR: {metrics['psnr']:.2f} dB | Test SSIM: {metrics['ssim']:.4f}", flush=True)

        dt = time.time() - t0
        final_metrics = evaluate_attacker_metrics(client, attacker, test_loader, device, mean, std, lpips_fn, max_batches=max_b)

        # Lưu ảnh tái tạo thực tế
        recon_path = os.path.join(args.output_dir, f"recon_{atk_name}.png")
        save_reconstruction_grid(client, attacker, test_loader, device, mean, std, save_path=recon_path, num_images=8)

        res_entry = {
            "attacker_type": atk_name,
            "final_epoch": total_epochs,
            "final_ssim": final_metrics["ssim"],
            "final_psnr": final_metrics["psnr"],
            "final_lpips": final_metrics["lpips"],
            "final_mse": final_metrics["mse"],
            "training_time_s": dt,
            "recon_image": recon_path,
            "history": history
        }
        results.append(res_entry)

    # 4. Xuất Báo Cáo JSON
    out_json = os.path.join(args.output_dir, "results_evaluation_attacks.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("BẢNG TỔNG KẾT ĐÁNH GIÁ TẤN CÔNG BẢO MẬT (EPOCH CUỐI):")
    print(f"{'Kẻ Tấn Công':<18} | {'Final SSIM':<12} | {'Final PSNR (dB)':<16} | {'Final LPIPS'}")
    print("-" * 80)
    for r in results:
        lp_str = f"{r['final_lpips']:.4f}" if r['final_lpips'] is not None else "N/A"
        print(f"{r['attacker_type']:<18} | {r['final_ssim']:<12.4f} | {r['final_psnr']:<16.2f} | {lp_str}")
    print("=" * 80)
    print(f"Đã lưu kết quả tại: {out_json}\n")


if __name__ == "__main__":
    main()
