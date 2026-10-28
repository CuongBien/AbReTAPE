#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_pcam_b0_attacks.py
==============================================================================
M2: FRESH ATTACKERS TRÊN CLIENT B0 CỦA PATCHCAMELYON (PCAM)
==============================================================================
Đánh giá tính bảo mật của biểu diễn trung gian smashed data z tại Client B0
trước 2 họ tấn công tái tạo thụ động (Passive Inversion Attack):
1. Fresh Conv Attacker: ConvDecoder tái tạo trực tiếp từ biểu diễn cut-layer (B, 64, 32, 32).
2. Fresh Learned MLP Attacker: LearnedAdaptiveDecoder (MLP Unprojector 65536 -> 512 -> 65536 + ConvDecoder).

Cấu hình kẻ tấn công chuẩn hóa như CIFAR:
- Client B0 đóng băng hoàn toàn (frozen, requires_grad=False, client.eval()).
- 50.000 ảnh huấn luyện PCAM (train_images.npy).
- 30 Epochs, Adam optimizer lr=1e-3, CosineAnnealingLR(T_max=30).
- Batch size: 128, Loss: MSELoss.
- Đánh giá trên toàn bộ 32,768 ảnh tập Test tại epoch cuối (không chọn checkpoint theo test).
- Ghi nhận: Test SSIM, Test PSNR (dB), Test MSE và xuất ảnh tái tạo 16 mẫu mô học.
==============================================================================
"""

import os
import sys
import time
import json
import csv
import argparse
import subprocess
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

from src.models import ClientModel, LearnedAdaptiveDecoder
from src.attacks.decoder import Decoder, TransposeLiftConvDecoder
from src.data.medical import get_pcam_dataloaders
from src.metrics.reconstruction import psnr_ssim, calculate_lpips, get_lpips_fn


class TeeLogger:
    """Tự động nhân đôi mọi luồng in (stdout/stderr) ra file log trên Google Drive."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        try:
            self.log_file.write(message)
            self.log_file.flush()
        except Exception:
            pass

    def flush(self):
        self.terminal.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass


def get_git_commit():
    """Lấy mã commit Git hiện tại."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "2e7acc160b5711652f4ae1b73c4d516086f68e0b"


def parse_args():
    parser = argparse.ArgumentParser(description="M2: Fresh Attackers trên Client B0 PCAM")
    parser.add_argument("--client-ckpt", type=str, default=None, help="Đường dẫn checkpoint client B0 (nếu None sẽ tìm theo seed)")
    parser.add_argument("--data-dir", type=str, default="./data/pcam_processed", help="Thư mục chứa mảng .npy tiền xử lý")
    parser.add_argument("--output-dir", type=str, default="./output/pcam_m2_b0_attacks", help="Thư mục xuất kết quả")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (42, 7, 2024)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện kẻ tấn công (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam (mặc định: 1e-3)")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Số chiều ẩn MLP unprojector (mặc định: 512)")
    parser.add_argument("--attackers", nargs="+", default=["conv", "mlp"], choices=["conv", "mlp"], help="Danh sách attacker cần đánh giá")
    parser.add_argument("--resume", action="store_true", help="Nạp checkpoint attacker đã lưu nếu có, bỏ qua huấn luyện lại")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm nhanh (1 epoch)")
    return parser.parse_args()


@torch.no_grad()
def evaluate_attacker(client, attacker, data_loader, device, mean, std, lpips_fn=None, max_batches=None):
    """Đánh giá PSNR, SSIM, MSE và LPIPS trên toàn bộ tập test."""
    client.eval()
    attacker.eval()

    total_samples = 0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_mse = 0.0
    criterion = nn.MSELoss()

    for idx, (x, _) in enumerate(data_loader):
        if max_batches is not None and idx >= max_batches:
            break
        x = x.to(device)
        b = x.size(0)

        zt = client(x)
        x_rec = attacker(zt)

        mse_val = criterion(x_rec, x).item()
        p, s = psnr_ssim(x, x_rec, mean, std)

        total_mse += mse_val * b
        total_psnr += p * b
        total_ssim += s * b

        if lpips_fn is not None:
            try:
                lp = calculate_lpips(x, x_rec, mean, std, lpips_fn)
                if lp is not None:
                    total_lpips += lp * b
            except Exception:
                pass

        total_samples += b

    return {
        "psnr": float(total_psnr / max(1, total_samples)),
        "ssim": float(total_ssim / max(1, total_samples)),
        "mse": float(total_mse / max(1, total_samples)),
        "lpips": float(total_lpips / max(1, total_samples)) if lpips_fn is not None else None,
        "num_samples": total_samples
    }


def save_pcam_reconstruction_grid(client, attacker, data_loader, device, mean, std, save_path, num_images=16):
    """Xuất lưới ảnh so sánh ảnh gốc PCAM vs ảnh tái tạo từ biểu diễn cut-layer."""
    client.eval()
    attacker.eval()

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            zt = client(x)
            x_rec = attacker(zt)
            break

    n = min(num_images, x.size(0))
    x = x[:n].cpu()
    x_rec = x_rec[:n].cpu()

    # Khôi phục giá trị pixel [0, 1] từ chuẩn hóa
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    x_orig = torch.clamp(x * s + m, 0.0, 1.0).numpy()
    x_recon = torch.clamp(x_rec * s + m, 0.0, 1.0).numpy()

    cols = min(8, n)
    rows = int(np.ceil(n / cols)) * 2

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)

    for i in range(n):
        r_orig = (i // cols) * 2
        r_rec = r_orig + 1
        c = i % cols

        img_orig = np.transpose(x_orig[i], (1, 2, 0))
        img_rec = np.transpose(x_recon[i], (1, 2, 0))

        axes[r_orig, c].imshow(img_orig)
        axes[r_orig, c].set_title(f"Gốc #{i+1}", fontsize=9)
        axes[r_orig, c].axis("off")

        axes[r_rec, c].imshow(img_rec)
        axes[r_rec, c].set_title(f"Tái tạo #{i+1}", fontsize=9)
        axes[r_rec, c].axis("off")

    # Tắt các ô trống nếu có
    for r in range(rows):
        for c in range(cols):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[TRỰC QUAN HÓA] Đã lưu lưới ảnh đối chứng tại: {save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Thiết lập file log
    log_file_p = os.path.join(args.output_dir, f"m2_attacks_pcam_b0_seed_{args.seed}.log")
    tee = TeeLogger(log_file_p)
    sys.stdout = tee
    sys.stderr = tee

    git_hash = get_git_commit()

    # Xác định đường dẫn Client Checkpoint
    if args.client_ckpt is not None:
        client_ckpt_p = args.client_ckpt
    else:
        # Tự động tìm checkpoint B0 tương ứng với seed
        candidate_paths = [
            f"./output/pcam_b0_vanilla/seed_{args.seed}/checkpoints/best_auc_b0_client.pt",
            f"./output/pcam_b0_vanilla/seed_{args.seed}/checkpoints/best_checkpoint.pt",
            f"/content/drive/MyDrive/pcam_b0_vanilla/seed_{args.seed}/checkpoints/best_auc_b0_client.pt",
            f"/content/drive/MyDrive/pcam_b0_vanilla/seed_{args.seed}/checkpoints/best_checkpoint.pt",
        ]
        client_ckpt_p = None
        for cp in candidate_paths:
            if os.path.isfile(cp):
                client_ckpt_p = cp
                break

    if client_ckpt_p is None or not os.path.isfile(client_ckpt_p):
        print(f"[LỖI] Không tìm thấy checkpoint Client B0 cho Seed {args.seed}!")
        print(f"Đã kiểm tra: {candidate_paths}")
        print("Vui lòng chỉ định rõ qua đối số --client-ckpt <đường_dẫn>.")
        sys.exit(1)

    print("=" * 80)
    print("M2: BẮT ĐẦU ĐÁNH GIÁ FRESH ATTACKERS TRÊN CLIENT B0 PCAM")
    print(f"Thiết bị: {device} | Seed: {args.seed} | Attacker Epochs: {args.epochs}")
    print(f"Client B0 Checkpoint: {client_ckpt_p}")
    print(f"Dataset Dir: {args.data_dir} | Output Dir: {args.output_dir}")
    print(f"Commit Git: {git_hash[:7]} ({git_hash})")
    print("=" * 80, flush=True)

    # 1. Nạp Dữ Liệu PCAM (Mean/Std từ pcam_stats.json)
    stats_p = os.path.join(args.data_dir, "pcam_stats.json")
    stats_dict = None
    if os.path.isfile(stats_p):
        with open(stats_p, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)

    train_loader, val_loader, test_loader, meta = get_pcam_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        stats=stats_dict
    )
    mean, std = meta["mean"], meta["std"]
    print(f"[DỮ LIỆU] Huấn luyện: {meta['num_train']:,} | Test: {meta['num_test']:,}")

    # 2. Khởi tạo và Nạp Client B0 (stem_stride=2 cho 64x64)
    client = ClientModel(in_channels=3, stem_stride=2).to(device)
    ckpt_raw = torch.load(client_ckpt_p, map_location=device, weights_only=False)
    if "client_state_dict" in ckpt_raw:
        client.load_state_dict(ckpt_raw["client_state_dict"])
    else:
        client.load_state_dict(ckpt_raw)

    client.eval()
    for p in client.parameters():
        p.requires_grad = False

    # Chốt chặn kiến trúc Client B0
    with torch.no_grad():
        dummy_in = torch.zeros(2, 3, 64, 64, device=device)
        dummy_z = client(dummy_in)
        assert dummy_z.shape == (2, 64, 32, 32), f"Sai shape client: {dummy_z.shape}"
        assert dummy_z.view(2, -1).shape[1] == 65536, f"Sai số chiều cut-layer D: {dummy_z.view(2, -1).shape[1]}"
    print(">>> [CHỐT CHẶN KIẾN TRÚC CLIENT B0 ĐẠT]: (64, 64) -> (64, 32, 32), D = 65,536 <<<\n")

    lpips_fn = get_lpips_fn(device=device)
    total_epochs = 1 if args.dry_run else args.epochs
    max_b = 2 if args.dry_run else None

    attack_results = {}

    # =========================================================================
    # HUẤN LUYỆN VÀ ĐÁNH GIÁ TỪNG KẺ TẤN CÔNG
    # =========================================================================
    for atk_type in args.attackers:
        print("\n" + "-" * 80)
        print(f"BẮT ĐẦU HUẤN LUYỆN FRESH ATTACKER: {atk_type.upper()}")
        print("-" * 80)

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        if atk_type == "conv":
            # Fresh Conv Attacker: Decoder 4 tầng + Upsample 2x (32x32 -> 64x64)
            attacker = TransposeLiftConvDecoder(
                bottleneck_weight=None,
                in_channels=64,
                out_channels=3,
                upscale_factor=2
            ).to(device)
            atk_desc = "Fresh Conv Lift Decoder (Conv 4 tầng + Bilinear Upsample 2x)"
        elif atk_type == "mlp":
            # Fresh Learned MLP Attacker: MLP Unprojector 65536 -> 512 -> 65536 + Conv Decoder 2x
            attacker = LearnedAdaptiveDecoder(
                k=65536,
                D=65536,
                hidden_dim=args.hidden_dim,
                out_channels=3,
                upscale_factor=2
            ).to(device)
            atk_desc = f"Fresh Learned MLP Attacker (Unprojector 65536 -> {args.hidden_dim} -> 65536 + ConvDecoder 2x)"
        else:
            continue

        print(f"Kiến trúc: {atk_desc}")
        atk_ckpt_p = os.path.join(args.output_dir, f"pcam_b0_{atk_type}_attacker_seed_{args.seed}.pt")
        history = []
        train_dur = 0.0

        if args.resume and os.path.isfile(atk_ckpt_p):
            print(f"\n[RESUME] Đã tìm thấy checkpoint attacker {atk_type.upper()} tại: {atk_ckpt_p}")
            print("Nạp trọng số và tiến hành đánh giá ngay trên tập test, không cần huấn luyện lại...")
            attacker.load_state_dict(torch.load(atk_ckpt_p, map_location=device, weights_only=False))
        else:
            optimizer = torch.optim.Adam(attacker.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
            criterion = nn.MSELoss()

            t0 = time.time()

            for ep in range(1, total_epochs + 1):
                attacker.train()
                train_mse_sum = 0.0
                n_batches = 0

                for b_idx, (x, _) in enumerate(train_loader):
                    if max_b is not None and b_idx >= max_b:
                        break
                    x = x.to(device)

                    with torch.no_grad():
                        zt = client(x)

                    x_rec = attacker(zt)
                    loss = criterion(x_rec, x)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_mse_sum += loss.item() * x.size(0)
                    n_batches += x.size(0)

                scheduler.step()
                train_mse = train_mse_sum / max(1, n_batches)

                # Đánh giá nhanh định kỳ hoặc epoch cuối
                if ep == 1 or ep % 5 == 0 or ep == total_epochs:
                    val_eval = evaluate_attacker(client, attacker, test_loader, device, mean, std, lpips_fn=None, max_batches=max_b)
                    print(
                        f"  [Ep {ep:2d}/{total_epochs:2d}] Train MSE: {train_mse:.4f} | "
                        f"Test PSNR: {val_eval['psnr']:5.2f} dB | Test SSIM: {val_eval['ssim']:.4f}",
                        flush=True
                    )
                    history.append({
                        "epoch": ep,
                        "train_mse": train_mse,
                        "test_psnr": val_eval["psnr"],
                        "test_ssim": val_eval["ssim"]
                    })

            train_dur = time.time() - t0
            print(f"\nHuấn luyện {atk_type.upper()} hoàn tất trong {train_dur/60:.1f} phút ({train_dur:.1f}s).")
            # Lưu checkpoint attacker để có thể resume hoặc tái sử dụng
            torch.save(attacker.state_dict(), atk_ckpt_p)
            print(f"Đã lưu checkpoint attacker {atk_type.upper()} tại: {atk_ckpt_p}")

        # Đánh giá toàn diện trên toàn bộ tập test tại epoch cuối
        print(f"Đánh giá chính thức {atk_type.upper()} tại Epoch cuối ({total_epochs})...")
        final_eval = evaluate_attacker(client, attacker, test_loader, device, mean, std, lpips_fn=lpips_fn, max_batches=max_b)

        # Xuất ảnh tái tạo
        grid_p = os.path.join(args.output_dir, f"recon_pcam_b0_{atk_type}_seed_{args.seed}.png")
        save_pcam_reconstruction_grid(client, attacker, test_loader, device, mean, std, grid_p, num_images=16)

        attack_results[atk_type] = {
            "attacker_type": atk_type,
            "description": atk_desc,
            "final_epoch": total_epochs,
            "test_ssim": final_eval["ssim"],
            "test_psnr": final_eval["psnr"],
            "test_mse": final_eval["mse"],
            "test_lpips": final_eval["lpips"],
            "training_time_s": train_dur,
            "history": history
        }

    # =========================================================================
    # LƯU KẾT QUẢ VÀ TỔNG HỢP JSON
    # =========================================================================
    summary_poc = {
        "dataset": "pcam",
        "method": "B0_Vanilla",
        "seed": args.seed,
        "client_ckpt": client_ckpt_p,
        "git_commit": git_hash[:7],
        "fresh_conv_ssim": round(attack_results.get("conv", {}).get("test_ssim", 0.0), 4),
        "fresh_conv_psnr": round(attack_results.get("conv", {}).get("test_psnr", 0.0), 2),
        "fresh_mlp_ssim": round(attack_results.get("mlp", {}).get("test_ssim", 0.0), 4),
        "fresh_mlp_psnr": round(attack_results.get("mlp", {}).get("test_psnr", 0.0), 2),
    }

    full_output = {
        "summary_for_poc": summary_poc,
        "attackers": attack_results
    }

    json_seed_p = os.path.join(args.output_dir, f"results_pcam_b0_attacks_seed_{args.seed}.json")
    with open(json_seed_p, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    csv_seed_p = os.path.join(args.output_dir, f"results_pcam_b0_attacks_seed_{args.seed}.csv")
    with open(csv_seed_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Seed", "Conv_SSIM", "Conv_PSNR", "MLP_SSIM", "MLP_PSNR", "Git_Commit"])
        writer.writerow([
            "B0_Vanilla",
            args.seed,
            f"{summary_poc['fresh_conv_ssim']:.4f}",
            f"{summary_poc['fresh_conv_psnr']:.2f}dB",
            f"{summary_poc['fresh_mlp_ssim']:.4f}",
            f"{summary_poc['fresh_mlp_psnr']:.2f}dB",
            summary_poc["git_commit"]
        ])

    print("\n" + "=" * 80)
    print(f"TỔNG KẾT M2 CHO CLIENT B0 PCAM (SEED {args.seed}):")
    print("=" * 80)
    print(f"  * Seed:                     {args.seed}")
    print(f"  * Fresh Conv Attacker SSIM: {summary_poc['fresh_conv_ssim']:.4f} (PSNR: {summary_poc['fresh_conv_psnr']:.2f} dB)")
    print(f"  * Fresh MLP Attacker SSIM:  {summary_poc['fresh_mlp_ssim']:.4f} (PSNR: {summary_poc['fresh_mlp_psnr']:.2f} dB)")
    print(f"  * File kết quả JSON:        {json_seed_p}")
    print("=" * 80 + "\n")

    tee.close()


if __name__ == "__main__":
    main()
