#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pcam_artape.py
==============================================================================
M4: HUẤN LUYỆN AR-TAPE (TASK-AWARE PRIVACY ENGINE) TRÊN PATCHCAMELYON (PCAM)
KÈM ĐÁNH GIÁ 4 HỌ TẤN CÔNG (FRESH CONV, FRESH MLP, FSHA gs=1.0 VÀ gs=5.0)
==============================================================================
Cấu hình chuẩn hóa M4:
- Kiến trúc Client: ResNet-18 Split (Stem stride 2, in_channels=3)
  Cut-layer tensor: (B, 64, 32, 32), D = 65,536
- AR-TAPE Defense: Mode F (fixed orthonormal bottleneck R in R^{65536 x 2048}, k=2048)
  Server unprojector: Linear Fixed (W_lift = R^T, 0 tham số thừa)
- Min-Max In-training Adversary: LearnedAdaptiveDecoder (2048 -> 512 -> 65536 + Conv 2x)
  Learning rate Adam: 1e-3, Cap = E[x_norm^2] = 1.0000
- Đối kháng: Lambda = 0.25, Warmup 10 epochs (tuyến tính từ 0 đến 0.25)
- Task Optimizer: SGD lr=0.1, momentum=0.9, weight_decay=5e-4, 100 epochs, CosineAnnealingLR
- Checkpoint: Lưu sau mỗi epoch, chọn checkpoint đánh giá Test theo Validation AUC cao nhất
- Đánh giá Kẻ tấn công (tại Epoch cuối 30, đánh giá trên frozen client):
  1. Fresh Conv Lift Attacker (W_lift = R^T, Bilinear Upsample 2x, 30 epochs)
  2. Fresh Learned MLP Attacker (2048 -> 512 -> 65536 + ConvDecoder 2x, 30 epochs)
  3. Active FSHA Attacker với grad_scale = 1.0 (30 epochs)
  4. Active FSHA Attacker với grad_scale = 5.0 (30 epochs)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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
    ClientWithBottleneck,
    ServerWithBottleneck,
    LearnedAdaptiveDecoder
)
from src.attacks.decoder import TransposeLiftConvDecoder
from src.attacks.fsha import (
    FSHADiscriminator,
    FSHAPilotAutoEncoder,
    fit_fsha_pilot
)
from src.data.medical import get_pcam_dataloaders
from src.metrics.clinical import compute_binary_clinical_metrics
from src.metrics.reconstruction import psnr_ssim, calculate_lpips, get_lpips_fn, denormalize


class TeeLogger:
    """Tự động ghi đồng thời ra màn hình terminal và file log."""
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
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "7bbaff0"


def parse_args():
    parser = argparse.ArgumentParser(description="M4: AR-TAPE (k=2048, lambda=0.25, cosine) trên PCAM kèm 4 Kẻ tấn công")
    parser.add_argument("--data-dir", type=str, default="./data/pcam_processed", help="Thư mục chứa mảng npy PCAM")
    parser.add_argument("--output-dir", type=str, default="./output/pcam_m4_artape", help="Thư mục lưu kết quả và checkpoint")
    parser.add_argument("--k", type=int, default=2048, help="Số chiều bottleneck (mặc định: 2048)")
    parser.add_argument("--m", type=int, default=512, help="Số chiều ẩn MLP (mặc định: 512)")
    parser.add_argument("--D", type=int, default=65536, help="Số chiều cut-layer (mặc định: 65536)")
    parser.add_argument("--lambda-val", type=float, default=0.25, help="Hệ số đối kháng lambda (mặc định: 0.25)")
    parser.add_argument("--cap", type=float, default=1.0000, help="Ngưỡng chặn MSE loss E[x_norm^2] (mặc định: 1.0000)")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Số epochs warmup lambda (mặc định: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (42, 7, 2024)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện Split Learning (mặc định: 100)")
    parser.add_argument("--attacker-epochs", type=int, default=30, help="Số epochs huấn luyện kẻ tấn công (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate SGD cho Task (mặc định: 0.1)")
    parser.add_argument("--adv-lr", type=float, default=1e-3, help="Learning rate Adam cho In-training Adversary (mặc định: 1e-3)")
    parser.add_argument("--attacker-lr", type=float, default=1e-3, help="Learning rate Adam cho Fresh Attackers (mặc định: 1e-3)")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint cuối nếu bị ngắt")
    parser.add_argument("--skip-attackers", action="store_true", help="Chỉ huấn luyện Task, bỏ qua bước đánh giá attacker")
    parser.add_argument("--eval-attackers-only", action="store_true", help="Chỉ đánh giá Attackers từ checkpoint có sẵn")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm nhanh (1 epoch)")
    return parser.parse_args()


@torch.no_grad()
def evaluate_split(client, server, data_loader, criterion, device, max_batches=None):
    """Đánh giá toàn diện mô hình Split Learning: Loss, Accuracy, AUC-ROC và Sens@90%Spec."""
    client.eval()
    server.eval()

    total_loss = 0.0
    all_targets = []
    all_probs = []

    for idx, (x, y) in enumerate(data_loader):
        if max_batches is not None and idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        z = client(x)
        logits = server(z)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        probs = F.softmax(logits, dim=1)[:, 1]

        all_targets.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    n_samples = len(all_targets)

    mean_loss = total_loss / max(1, n_samples)
    preds = (all_probs >= 0.5).astype(int)
    accuracy = (preds == all_targets).mean() * 100.0

    clin_metrics = compute_binary_clinical_metrics(all_targets, all_probs)
    auc_roc = clin_metrics.get("auc_roc", 0.5)
    return mean_loss, accuracy, auc_roc, clin_metrics


class FSHADecoderWrapper(nn.Module):
    """
    Wrapper kết nối Pilot Decoder trong FSHA để giải mã từ zt (B, k):
    zt (B, k) -> Server Lift (R^T) -> zh (B, 64, 32, 32) -> Pilot Decoder -> x_rec (B, 3, 64, 64)
    """
    def __init__(self, pilot_ae, server):
        super().__init__()
        self.pilot_ae = pilot_ae
        self.server = server

    def forward(self, zt):
        zh = self.server.lift(zt)
        zh_tensor = zh.view(zt.size(0), 64, 32, 32)
        return self.pilot_ae.decode(zh_tensor)


@torch.no_grad()
def evaluate_reconstruction_attacker(client, attacker, data_loader, device, mean, std, lpips_fn=None, max_batches=None):
    """Đánh giá PSNR, SSIM, MSE và LPIPS của kẻ tấn công tái tạo."""
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


def save_reconstruction_sample_grid(client, attacker, data_loader, device, mean, std, save_path, num_images=16, is_fsha=False, fsha_pilot=None):
    """Xuất lưới ảnh 16 mẫu so sánh ảnh mô học gốc và ảnh tái tạo."""
    client.eval()
    attacker.eval()

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            zt = client(x)
            if is_fsha and fsha_pilot is not None:
                zh = attacker(zt)
                x_rec = fsha_pilot.decode(zh)
            else:
                x_rec = attacker(zt)
            break

    n = min(num_images, x.size(0))
    x = x[:n].cpu()
    x_rec = x_rec[:n].cpu()

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
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file_p = os.path.join(args.output_dir, f"train_pcam_artape_seed_{args.seed}.log")
    tee = TeeLogger(log_file_p)
    sys.stdout = tee
    sys.stderr = tee

    git_hash = get_git_commit()

    print("=" * 90)
    print("M4: BẮT ĐẦU HUẤN LUYỆN AR-TAPE (TASK-AWARE PRIVACY ENGINE) TRÊN PCAM")
    print(f"Thiết bị: {device} | Seed: {args.seed} | Epochs: {args.epochs}")
    print(f"Cấu hình: k={args.k} | m={args.m} | D={args.D} | λ={args.lambda_val} | Cap={args.cap:.4f} | Warmup={args.warmup_epochs}")
    print(f"Dataset Dir: {args.data_dir} | Output Dir: {args.output_dir}")
    print(f"Commit Git: {git_hash[:7]} ({git_hash})")
    print("=" * 90, flush=True)

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
    print(f"[DỮ LIỆU] Huấn luyện: {meta['num_train']:,} | Val: {meta['num_val']:,} | Test: {meta['num_test']:,}")

    # Đo E[x^2] thực nghiệm trên tập Train đã chuẩn hóa
    clean_sq_sum = 0.0
    n_clean = 0
    with torch.no_grad():
        for b_idx, (x_chk, _) in enumerate(train_loader):
            if b_idx >= 20:
                break
            clean_sq_sum += (x_chk ** 2).sum().item()
            n_clean += x_chk.numel()
    measured_e_x2 = clean_sq_sum / max(n_clean, 1)
    applied_cap = float(args.cap)
    print(f"[CHỐT CHẶN CAP] E[x_norm^2] đo thực nghiệm trên batch train: {measured_e_x2:.4f} | Cap áp dụng: {applied_cap:.4f}\n")

    # 2. Khởi tạo Mô hình AR-TAPE (Client + Server + In-training Adversary)
    # Mode F: fixed_bottleneck=True, unprojector_mode="linear_fixed"
    client = ClientWithBottleneck(
        k=args.k,
        D=args.D,
        init_ortho=True,
        seed=args.seed,
        fixed_bottleneck=True,
        in_channels=3,
        stem_stride=2
    ).to(device)

    server = ServerWithBottleneck(
        k=args.k,
        m=args.m,
        D=args.D,
        num_classes=2,
        unprojector_mode="linear_fixed",
        ortho_matrix=client.R,
        seed=args.seed
    ).to(device)

    # In-training Adversary: 2048 -> 512 -> 65536 + Conv 2x (upscale_factor=2 cho 64x64)
    adv = LearnedAdaptiveDecoder(
        k=args.k,
        D=args.D,
        hidden_dim=args.m,
        upscale_factor=2
    ).to(device)

    # Chốt chặn kiến trúc
    with torch.no_grad():
        dummy_in = torch.zeros(2, 3, 64, 64, device=device)
        dummy_z = client(dummy_in)
        assert dummy_z.shape == (2, args.k), f"Sai shape bottleneck: {dummy_z.shape}, kỳ vọng (2, {args.k})"
        dummy_logits = server(dummy_z)
        assert dummy_logits.shape == (2, 2), f"Sai shape logits: {dummy_logits.shape}, kỳ vọng (2, 2)"
        dummy_rec = adv(dummy_z)
        assert dummy_rec.shape == (2, 3, 64, 64), f"Sai shape rec: {dummy_rec.shape}, kỳ vọng (2, 3, 64, 64)"
    print(f">>> [CHỐT CHẶN KIẾN TRÚC AR-TAPE ĐẠT]: (3, 64, 64) -> z_t ({args.k}) -> Logits (2) & x_rec (3, 64, 64) <<<\n")

    # 3. Optimizers & Schedulers
    # Task optimizer: SGD lr 0.1, momentum 0.9, weight_decay 5e-4
    task_params = [p for p in client.parameters() if p.requires_grad] + [p for p in server.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(task_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    total_epochs = 1 if args.dry_run else args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Adversary optimizer: Adam lr 1e-3
    opt_adv = torch.optim.Adam(adv.parameters(), lr=args.adv_lr)

    criterion_task = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    best_ckpt_p = os.path.join(ckpt_dir, "best_checkpoint.pt")
    last_ckpt_p = os.path.join(ckpt_dir, "last_checkpoint.pt")
    best_client_p = os.path.join(ckpt_dir, "best_auc_artape_client.pt")
    best_server_p = os.path.join(ckpt_dir, "best_auc_artape_server.pt")

    start_epoch = 1
    best_val_auc = 0.0
    best_epoch = 1
    history = []

    # 4. Phục hồi từ Checkpoint nếu có
    if args.resume and os.path.isfile(last_ckpt_p):
        print(f"[RESUME] Đang nạp checkpoint cuối từ: {last_ckpt_p}...")
        ckpt = torch.load(last_ckpt_p, map_location=device, weights_only=False)
        client.load_state_dict(ckpt["client_state_dict"])
        server.load_state_dict(ckpt["server_state_dict"])
        if "adv_state_dict" in ckpt:
            adv.load_state_dict(ckpt["adv_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "opt_adv_state_dict" in ckpt:
            opt_adv.load_state_dict(ckpt["opt_adv_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt.get("best_val_auc", 0.0)
        best_epoch = ckpt.get("best_epoch", 1)
        history = ckpt.get("history", [])
        print(f"[RESUME] Tiếp tục từ Epoch {start_epoch} (Best Epoch trước đó: {best_epoch}, Best Val AUC: {best_val_auc:.4f})")

    # =========================================================================
    # PHẦN 1: HUẤN LUYỆN MIN-MAX AR-TAPE (SPLIT LEARNING)
    # =========================================================================
    max_b = 2 if args.dry_run else None

    if not args.eval_attackers_only and start_epoch <= total_epochs:
        print("\n" + "-" * 90)
        print("BẮT ĐẦU HUẤN LUYỆN AR-TAPE (TASK + ADVERSARIAL LOSS WITH CAP & WARMUP)")
        print("-" * 90)
        t_start = time.time()

        for epoch in range(start_epoch, total_epochs + 1):
            ep_t0 = time.time()
            client.train()
            server.train()
            adv.train()

            train_loss_acc = 0.0
            train_correct = 0
            train_total = 0
            rec_mse_acc = 0.0

            # Tính lambda theo epoch với warmup tuyến tính
            curr_lam = args.lambda_val * min(1.0, epoch / max(1, args.warmup_epochs))

            for b_idx, (x, y) in enumerate(train_loader):
                if max_b is not None and b_idx >= max_b:
                    break
                x, y = x.to(device), y.to(device)

                # Forward Client (1 lần duy nhất)
                zt = client(x)

                # -------------------------------------------------------------
                # BƯỚC MAX: Cập nhật In-training Adversary
                # -------------------------------------------------------------
                zt_detached = zt.detach()
                x_rec = adv(zt_detached)
                loss_adv = criterion_mse(x_rec, x)

                opt_adv.zero_grad()
                loss_adv.backward()
                opt_adv.step()

                # -------------------------------------------------------------
                # BƯỚC MIN: Cập nhật Client + Server
                # -------------------------------------------------------------
                logits = server(zt)
                loss_task = criterion_task(logits, y)

                # Tính loss tái tạo của adversary trên zt có gradient về client
                adv.eval()
                for p in adv.parameters():
                    p.requires_grad = False

                x_rec_client = adv(zt)
                raw_rec_loss = criterion_mse(x_rec_client, x)
                rec_loss_capped = torch.clamp(raw_rec_loss, max=applied_cap)

                # AR-TAPE Objective: L_total = L_task - lambda * min(MSE, Cap)
                loss_total = loss_task - curr_lam * rec_loss_capped

                for p in adv.parameters():
                    p.requires_grad = True
                adv.train()

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                train_loss_acc += loss_task.item() * x.size(0)
                rec_mse_acc += raw_rec_loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == y).sum().item()
                train_total += x.size(0)

            scheduler.step()

            train_acc = (train_correct / max(train_total, 1)) * 100.0
            train_loss_mean = train_loss_acc / max(train_total, 1)
            train_rec_mean = rec_mse_acc / max(train_total, 1)

            val_loss, val_acc, val_auc, val_clin = evaluate_split(client, server, val_loader, criterion_task, device, max_batches=max_b)
            ep_dur = time.time() - ep_t0
            lr_curr = optimizer.param_groups[0]["lr"]

            print(
                f"  [Ep {epoch:3d}/{total_epochs:3d}] "
                f"Task Loss: {train_loss_mean:.4f} | Acc: {train_acc:6.2f}% | "
                f"Val AUC: {val_auc:.4f} (Best: {max(best_val_auc, val_auc):.4f}) | "
                f"Adv MSE: {train_rec_mean:.4f} | λ: {curr_lam:.3f} | "
                f"LR: {lr_curr:.5f} ({ep_dur:.1f}s)",
                flush=True
            )

            history.append({
                "epoch": epoch,
                "lr": lr_curr,
                "lambda": curr_lam,
                "train_task_loss": train_loss_mean,
                "train_acc": train_acc,
                "train_adv_mse": train_rec_mean,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_clinical": val_clin
            })

            # Lưu checkpoint định kỳ sau mỗi epoch
            torch.save({
                "epoch": epoch,
                "client_state_dict": client.state_dict(),
                "server_state_dict": server.state_dict(),
                "adv_state_dict": adv.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "opt_adv_state_dict": opt_adv.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "best_val_auc": max(best_val_auc, val_auc),
                "best_epoch": best_epoch if val_auc <= best_val_auc else epoch,
                "history": history
            }, last_ckpt_p)

            # Giữ riêng bản checkpoint có AUC Val cao nhất
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch

                torch.save(client.state_dict(), best_client_p)
                torch.save(server.state_dict(), best_server_p)
                torch.save({
                    "epoch": epoch,
                    "client_state_dict": client.state_dict(),
                    "server_state_dict": server.state_dict(),
                    "adv_state_dict": adv.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch
                }, best_ckpt_p)
                print(f"      >>> [BEST AUC] Đã lưu mô hình AR-TAPE tối ưu tại Epoch {epoch}: Val AUC = {val_auc:.4f} <<<")

        task_time = time.time() - t_start
        print(f"\nHuấn luyện AR-TAPE hoàn tất trong {task_time/60:.1f} phút ({task_time:.1f}s)!")
        print(f"Tối ưu tại Epoch {best_epoch} với Best Validation AUC: {best_val_auc:.4f}")
    else:
        task_time = 0.0

    # =========================================================================
    # PHẦN 2: ĐÁNH GIÁ CHÍNH THỨC TASK TRÊN TẬP TEST (CHECKPOINT VAL AUC TỐI ƯU)
    # =========================================================================
    if os.path.isfile(best_ckpt_p):
        ckpt_best = torch.load(best_ckpt_p, map_location=device, weights_only=False)
        client.load_state_dict(ckpt_best["client_state_dict"])
        server.load_state_dict(ckpt_best["server_state_dict"])
        best_epoch = ckpt_best.get("epoch", 1)
        best_val_auc = ckpt_best.get("val_auc", 0.0)
        # Đảm bảo best_client_p và best_server_p luôn sẵn sàng trên disk
        if not os.path.isfile(best_client_p):
            torch.save(client.state_dict(), best_client_p)
        if not os.path.isfile(best_server_p):
            torch.save(server.state_dict(), best_server_p)
        print(f"\n[TEST EVAL] Đã nạp thành công trọng số AR-TAPE tối ưu từ Epoch {best_epoch} (Val AUC: {best_val_auc:.4f}).")

    criterion_eval = nn.CrossEntropyLoss()
    test_loss, test_acc, test_auc, test_clin = evaluate_split(client, server, test_loader, criterion_eval, device, max_batches=max_b)
    test_sens_90 = float(test_clin.get("sensitivity_at_90_specificity", 0.0))

    print(f"\nKẾT QUẢ ĐỐI CHUẨN TASK AR-TAPE TRÊN PCAM (SEED {args.seed}):")
    print(f"  - Best Epoch (theo Val AUC): {best_epoch}")
    print(f"  - Best Val AUC-ROC:         {best_val_auc:.4f}")
    print(f"  - Test Accuracy:             {test_acc:.2f}%")
    print(f"  - Test AUC-ROC:              {test_auc:.4f}")
    print(f"  - Sens @ 90% Specificity:    {test_sens_90*100:.2f}% ({test_sens_90:.4f})")

    # =========================================================================
    # PHẦN 3: ĐÁNH GIÁ 4 HỌ TẤN CÔNG (FRESH CONV, FRESH MLP, FSHA gs=1.0 & gs=5.0)
    # =========================================================================
    attack_results = {}

    if not args.skip_attackers:
        print("\n" + "=" * 90)
        print(f"BẮT ĐẦU ĐÁNH GIÁ 4 KẺ TẤN CÔNG TRÊN CLIENT AR-TAPE (SEED {args.seed})")
        print(f"Client đã đóng băng. Các kẻ tấn công được huấn luyện 30 epochs (đánh giá tại epoch cuối).")
        print("=" * 90)

        client.eval()
        for p in client.parameters():
            p.requires_grad = False

        lpips_fn = get_lpips_fn(device=device)
        total_atk_epochs = 1 if args.dry_run else args.attacker_epochs

        # ---------------------------------------------------------------------
        # 3.1. FRESH CONV LIFT ATTACKER & FRESH LEARNED MLP ATTACKER
        # ---------------------------------------------------------------------
        passive_attackers = [
            ("conv", "Fresh Conv Lift Decoder (W_lift = R^T, Bilinear Upsample 2x)",
             TransposeLiftConvDecoder(bottleneck_weight=client.get_bottleneck_weight(), upscale_factor=2).to(device)),
            ("mlp", f"Fresh Learned MLP Attacker (Unprojector {args.k} -> {args.m} -> {args.D} + ConvDecoder 2x)",
             LearnedAdaptiveDecoder(k=args.k, D=args.D, hidden_dim=args.m, upscale_factor=2).to(device))
        ]

        for atk_name, atk_desc, attacker in passive_attackers:
            print("\n" + "-" * 80)
            print(f"Huấn luyện: {atk_name.upper()} - {atk_desc}")
            print("-" * 80)

            atk_ckpt_p = os.path.join(args.output_dir, f"pcam_artape_{atk_name}_attacker_seed_{args.seed}.pt")
            atk_hist = []
            atk_dur = 0.0

            if args.resume and os.path.isfile(atk_ckpt_p):
                print(f"[RESUME] Đã tìm thấy checkpoint attacker {atk_name.upper()} tại: {atk_ckpt_p}")
                print("Nạp trọng số và tiến hành đánh giá ngay trên tập test, không cần huấn luyện lại...")
                attacker.load_state_dict(torch.load(atk_ckpt_p, map_location=device, weights_only=False))
            else:
                atk_opt = torch.optim.Adam(attacker.parameters(), lr=args.attacker_lr)
                atk_sched = CosineAnnealingLR(atk_opt, T_max=total_atk_epochs)
                atk_crit = nn.MSELoss()

                t_atk0 = time.time()

                for a_ep in range(1, total_atk_epochs + 1):
                    attacker.train()
                    mse_accum = 0.0
                    samples_cnt = 0

                    for b_idx, (x, _) in enumerate(train_loader):
                        if max_b is not None and b_idx >= max_b:
                            break
                        x = x.to(device)

                        with torch.no_grad():
                            zt = client(x)

                        x_rec = attacker(zt)
                        loss = atk_crit(x_rec, x)

                        atk_opt.zero_grad()
                        loss.backward()
                        atk_opt.step()

                        mse_accum += loss.item() * x.size(0)
                        samples_cnt += x.size(0)

                    atk_sched.step()
                    ep_train_mse = mse_accum / max(1, samples_cnt)

                    if a_ep == 1 or a_ep % 5 == 0 or a_ep == total_atk_epochs:
                        eval_res = evaluate_reconstruction_attacker(client, attacker, test_loader, device, mean, std, lpips_fn=None, max_batches=max_b)
                        print(
                            f"  [Atk Ep {a_ep:2d}/{total_atk_epochs:2d}] Train MSE: {ep_train_mse:.4f} | "
                            f"Test PSNR: {eval_res['psnr']:5.2f} dB | Test SSIM: {eval_res['ssim']:.4f}",
                            flush=True
                        )
                        atk_hist.append({
                            "epoch": a_ep,
                            "train_mse": ep_train_mse,
                            "test_psnr": eval_res["psnr"],
                            "test_ssim": eval_res["ssim"]
                        })

                atk_dur = time.time() - t_atk0
                print(f"Huấn luyện {atk_name.upper()} hoàn tất trong {atk_dur/60:.1f} phút ({atk_dur:.1f}s).")
                torch.save(attacker.state_dict(), atk_ckpt_p)
                print(f"Đã lưu checkpoint attacker {atk_name.upper()} tại: {atk_ckpt_p}")

            # Đánh giá chính thức tại epoch cuối
            final_atk_eval = evaluate_reconstruction_attacker(client, attacker, test_loader, device, mean, std, lpips_fn=lpips_fn, max_batches=max_b)

            # Xuất lưới ảnh 16 mẫu mô học tái tạo
            grid_p = os.path.join(args.output_dir, f"recon_pcam_artape_{atk_name}_seed_{args.seed}.png")
            save_reconstruction_sample_grid(client, attacker, test_loader, device, mean, std, grid_p, num_images=16)

            attack_results[atk_name] = {
                "attacker_type": atk_name,
                "description": atk_desc,
                "final_epoch": total_atk_epochs,
                "test_ssim": final_atk_eval["ssim"],
                "test_psnr": final_atk_eval["psnr"],
                "test_mse": final_atk_eval["mse"],
                "test_lpips": final_atk_eval["lpips"],
                "training_time_s": atk_dur,
                "history": atk_hist
            }

        # ---------------------------------------------------------------------
        # 3.2. ACTIVE FSHA ATTACKERS (grad_scale = 1.0 và grad_scale = 5.0)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("CHUẨN BỊ ĐÁNH GIÁ TẤN CÔNG CHỦ ĐỘNG FSHA (FEATURE SPACE HIJACKING ATTACK)")
        print("=" * 80)

        # Phân chia dữ liệu FSHA: 25k private train (client) và 25k public train (pilot AE)
        train_ds_all = train_loader.dataset
        n_all_train = len(train_ds_all)
        n_priv = n_all_train // 2
        g_split = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_all_train, generator=g_split).tolist()
        priv_set = Subset(train_ds_all, perm[:n_priv])
        pub_set = Subset(train_ds_all, perm[n_priv:])

        use_cuda = torch.cuda.is_available()
        fsha_priv_loader = DataLoader(priv_set, batch_size=args.batch_size, shuffle=True, pin_memory=use_cuda)
        fsha_pub_loader = DataLoader(pub_set, batch_size=args.batch_size, shuffle=True, pin_memory=use_cuda)

        # 1. Huấn luyện Pilot AutoEncoder trên 25k ảnh Public (có cơ chế cache checkpoint)
        pilot_ae = FSHAPilotAutoEncoder(
            in_channels=3,
            latent_channels=64,
            stem_stride=2,
            upscale_factor=2
        ).to(device)

        pilot_ckpt_p = os.path.join(args.output_dir, f"pcam_artape_pilot_ae_seed_{args.seed}.pt")
        if args.resume and os.path.isfile(pilot_ckpt_p):
            print(f"[RESUME] Đã tìm thấy checkpoint Pilot AutoEncoder tại: {pilot_ckpt_p}. Nạp trọng số...")
            pilot_ae.load_state_dict(torch.load(pilot_ckpt_p, map_location=device, weights_only=False))
        else:
            print("[FSHA] Huấn luyện Pilot AutoEncoder trên tập dữ liệu Public (10 epochs)...")
            fit_fsha_pilot(
                pilot_ae=pilot_ae,
                pilot_server=None,
                pub_loader=fsha_pub_loader,
                test_loader=test_loader,
                epochs=10,
                lr=1e-3,
                device=device,
                mean=mean,
                std=std
            )
            torch.save(pilot_ae.state_dict(), pilot_ckpt_p)
            print(f"[FSHA] Pilot AutoEncoder đã hội tụ và lưu tại: {pilot_ckpt_p}")

        for gs in [1.0, 5.0]:
            fsha_name = f"fsha_gs{gs}"
            fsha_desc = f"FSHA Active Hijacking Attack (grad_scale={gs}, 30 epochs)"
            print("\n" + "-" * 80)
            print(f"Bắt đầu: {fsha_name.upper()} - {fsha_desc}")
            print("-" * 80)

            # Phục hồi Client và Server về checkpoint sạch (Best Val AUC) trước khi chạy mỗi kịch bản Hijacking
            if os.path.isfile(best_client_p):
                client.load_state_dict(torch.load(best_client_p, map_location=device, weights_only=False))
            elif os.path.isfile(best_ckpt_p):
                ckpt_b = torch.load(best_ckpt_p, map_location=device, weights_only=False)
                client.load_state_dict(ckpt_b["client_state_dict"])

            if os.path.isfile(best_server_p):
                server.load_state_dict(torch.load(best_server_p, map_location=device, weights_only=False))
            elif os.path.isfile(best_ckpt_p):
                ckpt_b = torch.load(best_ckpt_p, map_location=device, weights_only=False)
                server.load_state_dict(ckpt_b["server_state_dict"])

            fsha_ckpt_p = os.path.join(args.output_dir, f"pcam_artape_{fsha_name}_seed_{args.seed}.pt")
            fsha_hist = []
            fsha_dur = 0.0

            # Khởi tạo Discriminator (in_channels=64 cho tensor 64x32x32) và FSHA Decoder Wrapper
            discriminator = FSHADiscriminator(in_channels=64).to(device)
            fsha_decoder = FSHADecoderWrapper(pilot_ae=pilot_ae, server=server).to(device)

            if args.resume and os.path.isfile(fsha_ckpt_p):
                print(f"[RESUME] Đã tìm thấy checkpoint FSHA tại: {fsha_ckpt_p}. Nạp và đánh giá...")
                fsha_ckpt_data = torch.load(fsha_ckpt_p, map_location=device, weights_only=False)
                if "client" in fsha_ckpt_data:
                    client.load_state_dict(fsha_ckpt_data["client"])
                if "server" in fsha_ckpt_data:
                    server.load_state_dict(fsha_ckpt_data["server"])
                if "discriminator" in fsha_ckpt_data:
                    discriminator.load_state_dict(fsha_ckpt_data["discriminator"])
            else:
                client.train()
                for p in client.parameters():
                    p.requires_grad = True

                server.train()
                for p in server.parameters():
                    p.requires_grad = True

                opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
                opt_c = torch.optim.SGD(client.parameters(), lr=1e-4, momentum=0.9)
                opt_s = torch.optim.SGD(server.parameters(), lr=1e-3, momentum=0.9)
                criterion_task_fsha = nn.CrossEntropyLoss()

                t_fsha0 = time.time()

                for f_ep in range(1, total_atk_epochs + 1):
                    discriminator.train()
                    client.train()
                    server.train()

                    d_loss_acc = 0.0
                    n_f_samples = 0

                    pub_iter = iter(fsha_pub_loader)
                    for b_idx, (x_priv, y_priv) in enumerate(fsha_priv_loader):
                        if max_b is not None and b_idx >= max_b:
                            break
                        try:
                            x_pub, _ = next(pub_iter)
                        except StopIteration:
                            pub_iter = iter(fsha_pub_loader)
                            x_pub, _ = next(pub_iter)

                        x_priv, y_priv = x_priv.to(device), y_priv.to(device)
                        x_pub = x_pub.to(device)

                        # 1. Update Discriminator: z_pilot vs z_client (sau khi nâng chiều W_lift = R^T về 64x32x32)
                        opt_d.zero_grad()
                        with torch.no_grad():
                            zt_priv = client(x_priv)
                            zh_priv = server.lift(zt_priv).view(zt_priv.size(0), 64, 32, 32)
                            z_pub, _ = pilot_ae(x_pub)

                        d_real = discriminator(z_pub)
                        d_fake = discriminator(zh_priv)

                        loss_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
                        loss_d.backward()
                        opt_d.step()

                        # 2. Adversarial Gradient Hijacking vào Client & thích ứng Server
                        opt_c.zero_grad()
                        opt_s.zero_grad()

                        zt_priv = client(x_priv)
                        zh_priv = server.lift(zt_priv).view(zt_priv.size(0), 64, 32, 32)

                        # Task loss hợp pháp từ Server
                        logits = server.base_server(zh_priv)
                        loss_task = criterion_task_fsha(logits, y_priv)

                        # Adversarial hijack loss
                        d_fake_adv = discriminator(zh_priv)
                        loss_adv = - d_fake_adv.mean()

                        loss_hijack = loss_task + gs * loss_adv
                        loss_hijack.backward()

                        opt_c.step()
                        opt_s.step()

                        d_loss_acc += loss_d.item() * x_priv.size(0)
                        n_f_samples += x_priv.size(0)

                    ep_d_loss = d_loss_acc / max(1, n_f_samples)

                    if f_ep == 1 or f_ep % 5 == 0 or f_ep == total_atk_epochs:
                        # Đánh giá nhanh FSHA
                        client.eval()
                        server.eval()
                        fsha_decoder.eval()
                        pilot_ae.eval()

                        fsha_eval = evaluate_reconstruction_attacker(
                            client, fsha_decoder, test_loader, device, mean, std, lpips_fn=None, max_batches=max_b
                        )
                        # Đánh giá lại task acc bị ảnh hưởng bởi gradient hijacking
                        _, fsha_task_acc, _, _ = evaluate_split(client, server, test_loader, criterion_eval, device, max_batches=max_b)

                        print(
                            f"  [FSHA Ep {f_ep:2d}/{total_atk_epochs:2d}] D Loss: {ep_d_loss:.4f} | "
                            f"Task Acc: {fsha_task_acc:5.2f}% | Test PSNR: {fsha_eval['psnr']:5.2f} dB | Test SSIM: {fsha_eval['ssim']:.4f}",
                            flush=True
                        )
                        fsha_hist.append({
                            "epoch": f_ep,
                            "d_loss": ep_d_loss,
                            "task_acc": fsha_task_acc,
                            "test_psnr": fsha_eval["psnr"],
                            "test_ssim": fsha_eval["ssim"]
                        })

                fsha_dur = time.time() - t_fsha0
                print(f"Huấn luyện {fsha_name.upper()} hoàn tất trong {fsha_dur/60:.1f} phút ({fsha_dur:.1f}s).")
                torch.save({
                    "client": client.state_dict(),
                    "server": server.state_dict(),
                    "discriminator": discriminator.state_dict()
                }, fsha_ckpt_p)
                print(f"Đã lưu checkpoint {fsha_name.upper()} tại: {fsha_ckpt_p}")

            # Đánh giá chính thức FSHA tại epoch cuối 30
            client.eval()
            server.eval()
            fsha_decoder.eval()
            pilot_ae.eval()
            final_fsha_eval = evaluate_reconstruction_attacker(
                client, fsha_decoder, test_loader, device, mean, std, lpips_fn=lpips_fn, max_batches=max_b
            )
            _, final_fsha_acc, _, _ = evaluate_split(client, server, test_loader, criterion_eval, device, max_batches=max_b)

            grid_p = os.path.join(args.output_dir, f"recon_pcam_artape_{fsha_name}_seed_{args.seed}.png")
            save_reconstruction_sample_grid(client, fsha_decoder, test_loader, device, mean, std, grid_p, num_images=16)

            attack_results[fsha_name] = {
                "attacker_type": fsha_name,
                "grad_scale": gs,
                "description": fsha_desc,
                "final_epoch": total_atk_epochs,
                "converged_task_acc": final_fsha_acc,
                "test_ssim": final_fsha_eval["ssim"],
                "test_psnr": final_fsha_eval["psnr"],
                "test_mse": final_fsha_eval["mse"],
                "test_lpips": final_fsha_eval["lpips"],
                "training_time_s": fsha_dur,
                "history": fsha_hist
            }

    # =========================================================================
    # PHẦN 4: TỔNG HỢP VÀ XUẤT KẾT QUẢ JSON & CSV
    # =========================================================================
    conv_ssim = attack_results.get("conv", {}).get("test_ssim", 0.0)
    conv_psnr = attack_results.get("conv", {}).get("test_psnr", 0.0)
    mlp_ssim = attack_results.get("mlp", {}).get("test_ssim", 0.0)
    mlp_psnr = attack_results.get("mlp", {}).get("test_psnr", 0.0)
    fsha1_ssim = attack_results.get("fsha_gs1.0", {}).get("test_ssim", 0.0)
    fsha1_acc = attack_results.get("fsha_gs1.0", {}).get("converged_task_acc", 0.0)
    fsha5_ssim = attack_results.get("fsha_gs5.0", {}).get("test_ssim", 0.0)
    fsha5_acc = attack_results.get("fsha_gs5.0", {}).get("converged_task_acc", 0.0)

    summary_poc = {
        "dataset": "pcam",
        "method": "AR_TAPE",
        "k": args.k,
        "m": args.m,
        "lambda": args.lambda_val,
        "cap": applied_cap,
        "warmup_epochs": args.warmup_epochs,
        "seed": args.seed,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(round(best_val_auc, 4)),
        "test_auc": float(round(test_auc, 4)),
        "test_accuracy": float(round(test_acc, 2)),
        "test_sens_at_90_spec": float(round(test_sens_90, 4)),
        "fresh_conv_ssim": float(round(conv_ssim, 4)),
        "fresh_conv_psnr": float(round(conv_psnr, 2)),
        "fresh_mlp_ssim": float(round(mlp_ssim, 4)),
        "fresh_mlp_psnr": float(round(mlp_psnr, 2)),
        "fsha_gs1_ssim": float(round(fsha1_ssim, 4)),
        "fsha_gs1_task_acc": float(round(fsha1_acc, 2)),
        "fsha_gs5_ssim": float(round(fsha5_ssim, 4)),
        "fsha_gs5_task_acc": float(round(fsha5_acc, 2)),
        "git_commit": git_hash[:7],
    }

    full_results = {
        "summary_for_poc": summary_poc,
        "clinical_metrics": test_clin,
        "attackers": attack_results
    }

    json_seed_p = os.path.join(args.output_dir, f"results_pcam_artape_seed_{args.seed}.json")
    with open(json_seed_p, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    csv_seed_p = os.path.join(args.output_dir, f"results_pcam_artape_seed_{args.seed}.csv")
    with open(csv_seed_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "k", "lambda", "Cap", "Seed", "Val_AUC", "Test_AUC", "Test_Acc", "Sens@90%Spec", "Conv_SSIM", "Conv_PSNR", "MLP_SSIM", "MLP_PSNR", "FSHA_gs1_SSIM", "FSHA_gs5_SSIM", "Git_Commit"])
        writer.writerow([
            "AR_TAPE",
            args.k,
            args.lambda_val,
            f"{applied_cap:.4f}",
            args.seed,
            f"{summary_poc['best_val_auc']:.4f}",
            f"{summary_poc['test_auc']:.4f}",
            f"{summary_poc['test_accuracy']:.2f}%",
            f"{summary_poc['test_sens_at_90_spec']*100:.2f}%",
            f"{summary_poc['fresh_conv_ssim']:.4f}",
            f"{summary_poc['fresh_conv_psnr']:.2f}dB",
            f"{summary_poc['fresh_mlp_ssim']:.4f}",
            f"{summary_poc['fresh_mlp_psnr']:.2f}dB",
            f"{summary_poc['fsha_gs1_ssim']:.4f}",
            f"{summary_poc['fsha_gs5_ssim']:.4f}",
            summary_poc["git_commit"]
        ])

    print("\n" + "=" * 90)
    print(f"TỔNG KẾT M4 AR-TAPE CHO PCAM (SEED {args.seed}):")
    print("=" * 90)
    print(f"  * Seed:                    {args.seed}")
    print(f"  * Best Epoch (Val AUC):    {best_epoch} (Val AUC: {best_val_auc:.4f})")
    print(f"  * Test Accuracy:           {test_acc:.2f}%")
    print(f"  * Test AUC-ROC:            {test_auc:.4f}")
    print(f"  * Sens @ 90% Spec:         {test_sens_90*100:.2f}%")
    print(f"  * Fresh Conv Attacker:     SSIM = {conv_ssim:.4f} | PSNR = {conv_psnr:.2f} dB")
    print(f"  * Fresh MLP Attacker:      SSIM = {mlp_ssim:.4f} | PSNR = {mlp_psnr:.2f} dB")
    print(f"  * Active FSHA (gs=1.0):    SSIM = {fsha1_ssim:.4f} | Task Acc = {fsha1_acc:.2f}%")
    print(f"  * Active FSHA (gs=5.0):    SSIM = {fsha5_ssim:.4f} | Task Acc = {fsha5_acc:.2f}%")
    print(f"  * File kết quả JSON:       {json_seed_p}")
    print(f"  * File kết quả CSV:        {csv_seed_p}")
    print("=" * 90 + "\n")

    tee.close()


if __name__ == "__main__":
    main()
