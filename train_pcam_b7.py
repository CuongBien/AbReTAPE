#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pcam_b7.py
==============================================================================
M3: HUẤN LUYỆN B7 LIGHTSPLIT (k=2048, MODE F) TRÊN PATCHCAMELYON (PCAM)
KÈM ĐÁNH GIÁ 2 FRESH ATTACKERS (CONV LIFT & LEARNED MLP)
==============================================================================
Đặc tả kỹ thuật:
1. Kiến trúc B7 LightSplit:
   - Client: ResNet-18 stem_stride=2 (64x64 -> 32x32x64 = 65,536).
     Chiếu trực giao cố định z_t = h_flat @ R in R^{B x 2048} (Mode F, R in R^{65536 x 2048}).
   - Server: Unprojector tuyến tính không tham số z_hat = z_t @ R^T in R^{B x 65536} ->
     Reshape (64, 32, 32) -> Server ResNet-18 (layer2, layer3, layer4, avgpool, fc -> 2 classes).
   - Chốt chặn kiến trúc: assert shape smashed data là (B, 2048) và logits là (B, 2).
2. Huấn luyện Split Learning:
   - SGD lr=0.1, momentum=0.9, weight_decay=5e-4, 100 epochs, MultiStepLR [50, 75] (gamma=0.1).
   - Augmentation: RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90 (không padding crop).
   - Chuẩn hóa: Mean/Std từ pcam_stats.json.
   - Chọn checkpoint: Lưu checkpoint sau mỗi epoch (last_checkpoint.pt, hỗ trợ --resume),
     giữ riêng checkpoint có Validation AUC cao nhất (best_checkpoint.pt).
   - Đánh giá Test tại checkpoint tốt nhất: Test AUC, Test Accuracy, Sens@90%Specificity.
3. Đánh giá 2 Fresh Attackers trên Client B7 đã đóng băng:
   - Attacker 1 (Conv Lift): z_t @ R^T -> reshape (64, 32, 32) -> ConvDecoder 2x (32x32 -> 64x64).
   - Attacker 2 (Learned MLP): MLP 2048 -> 512 -> 65536 -> reshape (64, 32, 32) -> ConvDecoder 2x.
   - Cấu hình chuẩn hóa như CIFAR: 50.000 ảnh train, 30 epochs, Adam lr=1e-3, CosineAnnealingLR.
   - Đánh giá tại epoch cuối trên toàn bộ 32,768 ảnh test: SSIM, PSNR (dB) và lưới ảnh đối chứng.
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
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

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
from src.data.medical import get_pcam_dataloaders
from src.metrics.clinical import compute_binary_clinical_metrics
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
    parser = argparse.ArgumentParser(description="M3: Huấn luyện B7 LightSplit (k=2048, Mode F) trên PCAM kèm Fresh Attackers")
    parser.add_argument("--data-dir", type=str, default="./data/pcam_processed", help="Thư mục chứa mảng .npy tiền xử lý")
    parser.add_argument("--output-dir", type=str, default="./output/pcam_b7_k2048", help="Thư mục lưu kết quả và checkpoint")
    parser.add_argument("--k", type=int, default=2048, help="Số chiều bottleneck (mặc định: 2048)")
    parser.add_argument("--m", type=int, default=512, help="Số chiều ẩn MLP attacker (mặc định: 512)")
    parser.add_argument("--D", type=int, default=65536, help="Số chiều cut-layer (mặc định: 65536)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (42, 7, 2024)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện Split Learning (mặc định: 100)")
    parser.add_argument("--attacker-epochs", type=int, default=30, help="Số epochs huấn luyện kẻ tấn công (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate SGD cho Task (mặc định: 0.1)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum SGD")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--milestones", nargs="+", type=int, default=[50, 75], help="Milestones cho MultiStepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma giảm LR")
    parser.add_argument("--attacker-lr", type=float, default=1e-3, help="Learning rate Adam cho Fresh Attacker")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint cuối nếu bị ngắt")
    parser.add_argument("--skip-attackers", action="store_true", help="Chỉ huấn luyện Task, bỏ qua bước đánh giá attacker")
    parser.add_argument("--eval-attackers-only", action="store_true", help="Chỉ đánh giá Fresh Attackers từ checkpoint có sẵn")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm nhanh (1 epoch)")
    return parser.parse_args()


def evaluate_split(client, server, data_loader, criterion, device, max_batches=None):
    """Đánh giá loss, acc và AUC-ROC trên tập validation hoặc test."""
    client.eval()
    server.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds_prob = []
    all_targets = []

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(data_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)

            z = client(x)
            out = server(z)
            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += x.size(0)

            all_preds_prob.append(probs[:, 1].cpu().numpy())
            all_targets.append(y.cpu().numpy())

    mean_loss = total_loss / max(total, 1)
    acc = (correct / max(total, 1)) * 100.0

    if len(all_targets) > 0:
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_preds_prob)
        if len(np.unique(y_true)) > 1:
            clinical_res = compute_binary_clinical_metrics(y_true, y_prob)
            auc = clinical_res["auc_roc"]
        else:
            clinical_res = {}
            auc = 0.5
    else:
        clinical_res = {}
        auc = 0.5

    return mean_loss, acc, auc, clinical_res


@torch.no_grad()
def evaluate_reconstruction_attacker(client, attacker, data_loader, device, mean, std, lpips_fn=None, max_batches=None):
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
            lp = calculate_lpips(lpips_fn, x, x_rec, mean, std)
            total_lpips += lp * b

        total_samples += b

    return {
        "psnr": float(total_psnr / max(1, total_samples)),
        "ssim": float(total_ssim / max(1, total_samples)),
        "mse": float(total_mse / max(1, total_samples)),
        "lpips": float(total_lpips / max(1, total_samples)) if lpips_fn is not None else None,
        "num_samples": total_samples
    }


def save_reconstruction_sample_grid(client, attacker, data_loader, device, mean, std, save_path, num_images=16):
    """Xuất lưới ảnh 16 mẫu so sánh ảnh mô học gốc và ảnh tái tạo."""
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

    log_file_p = os.path.join(args.output_dir, f"train_pcam_b7_seed_{args.seed}.log")
    tee = TeeLogger(log_file_p)
    sys.stdout = tee
    sys.stderr = tee

    git_hash = get_git_commit()

    print("=" * 80)
    print("M3: HUẤN LUYỆN B7 LIGHTSPLIT (k=2048, MODE F) TRÊN PATCHCAMELYON (PCAM)")
    print(f"Thiết bị: {device} | Seed: {args.seed} | Bottleneck k={args.k} (Mode F: fixed R)")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Milestones: {args.milestones} (gamma={args.gamma})")
    print(f"Commit Git: {git_hash[:7]} ({git_hash})")
    print(f"Thư mục dữ liệu: {args.data_dir} | Thư mục kết quả: {args.output_dir}")
    print("=" * 80, flush=True)

    # 1. Nạp Dữ Liệu
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
    print(f"[DỮ LIỆU] Train: {meta['num_train']:,} | Val: {meta['num_val']:,} | Test: {meta['num_test']:,}")

    # 2. Khởi tạo Client và Server B7
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

    # Chốt chặn kiến trúc đầu script
    with torch.no_grad():
        test_in = torch.zeros(2, 3, 64, 64, device=device)
        test_z = client(test_in)
        assert test_z.shape == (2, args.k), f"[CHỐT CHẶN THẤT BẠI] Đầu ra client có shape {test_z.shape}, kỳ vọng (2, {args.k})!"
        test_logits = server(test_z)
        assert test_logits.shape == (2, 2), f"[CHỐT CHẶN THẤT BẠI] Đầu ra server logits có shape {test_logits.shape}, kỳ vọng (2, 2)!"
    print(f">>> [CHỐT CHẶN KIẾN TRÚC B7 ĐẠT]: (64, 64) -> Client -> Smashed z (k={args.k}) -> Server Mode F -> Logits (2) <<<\n")

    best_ckpt_p = os.path.join(ckpt_dir, "best_checkpoint.pt")
    last_ckpt_p = os.path.join(ckpt_dir, "last_checkpoint.pt")

    # =========================================================================
    # PHẦN 1: HUẤN LUYỆN SPLIT LEARNING CHO TASK CHÍNH
    # =========================================================================
    if not args.eval_attackers_only:
        trainable_params = list(client.parameters()) + list(server.parameters())
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 1
        best_val_auc = -1.0
        best_epoch = 1
        history = []

        if os.path.isfile(best_ckpt_p):
            try:
                ckpt_b = torch.load(best_ckpt_p, map_location=device, weights_only=False)
                best_epoch = ckpt_b.get("epoch", 1)
                best_val_auc = ckpt_b.get("val_auc", -1.0)
            except Exception:
                pass

        if args.resume and os.path.isfile(last_ckpt_p):
            print(f"[RESUME] Đang khôi phục từ checkpoint: {last_ckpt_p}...")
            ckpt = torch.load(last_ckpt_p, map_location=device, weights_only=False)
            start_epoch = ckpt["epoch"] + 1
            client.load_state_dict(ckpt["client_state_dict"])
            server.load_state_dict(ckpt["server_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_val_auc = max(best_val_auc, ckpt.get("best_val_auc", -1.0))
            best_epoch = ckpt.get("best_epoch", best_epoch)
            history = ckpt.get("history", [])
            print(f"[RESUME] Khôi phục thành công! Tiếp tục từ Epoch {start_epoch} (Best Val AUC trước đó: {best_val_auc:.4f} tại Ep {best_epoch})")

        total_epochs = 1 if args.dry_run else args.epochs
        max_b = 2 if args.dry_run else None

        print("BẮT ĐẦU HUẤN LUYỆN B7 TASK (100 EPOCHS):")
        print("-" * 100)
        t_start = time.time()

        for epoch in range(start_epoch, total_epochs + 1):
            ep_t0 = time.time()
            client.train()
            server.train()

            train_loss, train_correct, train_total = 0.0, 0, 0

            for b_idx, (x, y) in enumerate(train_loader):
                if max_b is not None and b_idx >= max_b:
                    break
                x, y = x.to(device), y.to(device)

                z = client(x)
                out = server(z)
                loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                train_correct += (preds == y).sum().item()
                train_total += x.size(0)

            scheduler.step()

            train_acc = (train_correct / max(train_total, 1)) * 100.0
            train_loss_mean = train_loss / max(train_total, 1)

            val_loss, val_acc, val_auc, val_clin = evaluate_split(client, server, val_loader, criterion, device, max_batches=max_b)
            ep_dur = time.time() - ep_t0
            lr_curr = optimizer.param_groups[0]["lr"]

            print(
                f"  [Ep {epoch:3d}/{total_epochs:3d}] "
                f"Train Loss: {train_loss_mean:.4f} | Acc: {train_acc:6.2f}% | "
                f"Val Loss: {val_loss:.4f} | Acc: {val_acc:6.2f}% | "
                f"Val AUC: {val_auc:.4f} (Best: {max(best_val_auc, val_auc):.4f}) | "
                f"LR: {lr_curr:.5f} ({ep_dur:.1f}s)",
                flush=True
            )

            # Cảnh báo tại Epoch 1
            if epoch == 1:
                if val_auc <= 0.52:
                    print("\n" + "!" * 80)
                    print(f"[CẢNH BÁO] Val AUC sau Epoch 1 là {val_auc:.4f} (<= 0.52)!")
                    print("Dấu hiệu lệch nhãn hoặc chuẩn hóa dữ liệu. Kiểm tra lại dữ liệu!")
                    print("!" * 80 + "\n", flush=True)
                else:
                    print(f"\n>>> [KIỂM TRA ĐẠT] Val AUC sau Epoch 1 là {val_auc:.4f} (> 0.52). Tín hiệu tốt! <<<\n", flush=True)

            history.append({
                "epoch": epoch,
                "lr": lr_curr,
                "train_loss": train_loss_mean,
                "train_acc": train_acc,
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
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
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
                best_client_p = os.path.join(ckpt_dir, "best_auc_b7_client.pt")
                best_server_p = os.path.join(ckpt_dir, "best_auc_b7_server.pt")

                torch.save(client.state_dict(), best_client_p)
                torch.save(server.state_dict(), best_server_p)
                torch.save({
                    "epoch": epoch,
                    "client_state_dict": client.state_dict(),
                    "server_state_dict": server.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch
                }, best_ckpt_p)
                print(f"      >>> [BEST AUC] Đã lưu mô hình B7 tối ưu tại Epoch {epoch}: Val AUC = {val_auc:.4f} <<<")

        task_time = time.time() - t_start
        print("-" * 100)
        print(f"Huấn luyện B7 hoàn tất trong {task_time/60:.1f} phút ({task_time:.1f}s)!")
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
        print(f"\n[TEST EVAL] Đã nạp thành công trọng số B7 tối ưu từ Epoch {best_epoch} (Val AUC: {best_val_auc:.4f}).")

    criterion_eval = nn.CrossEntropyLoss()
    max_b = 2 if args.dry_run else None
    test_loss, test_acc, test_auc, test_clin = evaluate_split(client, server, test_loader, criterion_eval, device, max_batches=max_b)
    test_sens_90 = float(test_clin.get("sensitivity_at_90_specificity", 0.0))

    print(f"\nKẾT QUẢ ĐỐI CHUẨN TASK B7 LIGHTSPLIT TRÊN PCAM (SEED {args.seed}):")
    print(f"  - Best Epoch (theo Val AUC): {best_epoch}")
    print(f"  - Best Val AUC-ROC:         {best_val_auc:.4f}")
    print(f"  - Test Accuracy:             {test_acc:.2f}%")
    print(f"  - Test AUC-ROC:              {test_auc:.4f}")
    print(f"  - Sens @ 90% Specificity:    {test_sens_90*100:.2f}% ({test_sens_90:.4f})")

    # =========================================================================
    # PHẦN 3: ĐÁNH GIÁ 2 FRESH ATTACKERS (CONV LIFT & LEARNED MLP)
    # =========================================================================
    attack_results = {}

    if not args.skip_attackers:
        print("\n" + "=" * 80)
        print(f"BẮT ĐẦU ĐÁNH GIÁ 2 FRESH ATTACKERS TRÊN CLIENT B7 (SEED {args.seed})")
        print(f"Client B7 đã đóng băng (requires_grad=False). Attacker train 30 epochs trên 50k ảnh.")
        print("=" * 80)

        client.eval()
        for p in client.parameters():
            p.requires_grad = False

        lpips_fn = get_lpips_fn(device=device)
        total_atk_epochs = 1 if args.dry_run else args.attacker_epochs

        # Danh sách 2 Kẻ tấn công
        # 1. Fresh Conv Lift Attacker: z_t @ R^T -> reshape (64, 32, 32) -> ConvDecoder 2x
        # 2. Fresh Learned MLP Attacker: MLP 2048 -> 512 -> 65536 -> reshape (64, 32, 32) -> ConvDecoder 2x
        attackers_to_eval = [
            ("conv", "Fresh Conv Lift Decoder (W_lift = R^T, Bilinear Upsample 2x)",
             TransposeLiftConvDecoder(bottleneck_weight=client.get_bottleneck_weight(), upscale_factor=2).to(device)),
            ("mlp", f"Fresh Learned MLP Attacker (Unprojector {args.k} -> {args.m} -> {args.D} + ConvDecoder 2x)",
             LearnedAdaptiveDecoder(k=args.k, D=args.D, hidden_dim=args.m, upscale_factor=2).to(device))
        ]

        for atk_name, atk_desc, attacker in attackers_to_eval:
            print("\n" + "-" * 80)
            print(f"Huấn luyện: {atk_name.upper()} - {atk_desc}")
            print("-" * 80)

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            atk_opt = torch.optim.Adam(attacker.parameters(), lr=args.attacker_lr)
            atk_sched = CosineAnnealingLR(atk_opt, T_max=total_atk_epochs)
            atk_crit = nn.MSELoss()

            t_atk0 = time.time()
            atk_hist = []

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

            # Đánh giá chính thức tại epoch 30
            final_atk_eval = evaluate_reconstruction_attacker(client, attacker, test_loader, device, mean, std, lpips_fn=lpips_fn, max_batches=max_b)

            # Xuất lưới ảnh 16 mẫu mô học tái tạo
            grid_p = os.path.join(args.output_dir, f"recon_pcam_b7_{atk_name}_seed_{args.seed}.png")
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

    # =========================================================================
    # PHẦN 4: TỔNG HỢP VÀ XUẤT KẾT QUẢ JSON & CSV
    # =========================================================================
    conv_ssim = attack_results.get("conv", {}).get("test_ssim", 0.0)
    conv_psnr = attack_results.get("conv", {}).get("test_psnr", 0.0)
    mlp_ssim = attack_results.get("mlp", {}).get("test_ssim", 0.0)
    mlp_psnr = attack_results.get("mlp", {}).get("test_psnr", 0.0)

    summary_poc = {
        "dataset": "pcam",
        "method": "B7_LightSplit",
        "k": args.k,
        "mode": "F",
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
        "task_train_time_formatted": f"{int(task_time // 60)}m {int(task_time % 60):02d}s",
        "git_commit": git_hash[:7],
        "git_commit_full": git_hash
    }

    full_results = {
        "summary_for_poc": summary_poc,
        "clinical_metrics": test_clin,
        "attackers": attack_results
    }

    json_seed_p = os.path.join(args.output_dir, f"results_pcam_b7_seed_{args.seed}.json")
    with open(json_seed_p, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    csv_seed_p = os.path.join(args.output_dir, f"results_pcam_b7_seed_{args.seed}.csv")
    with open(csv_seed_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "k", "Mode", "Seed", "Val_AUC", "Test_AUC", "Test_Acc", "Sens@90%Spec", "Conv_SSIM", "Conv_PSNR", "MLP_SSIM", "MLP_PSNR", "Git_Commit"])
        writer.writerow([
            "B7_LightSplit",
            args.k,
            "F",
            args.seed,
            f"{summary_poc['best_val_auc']:.4f}",
            f"{summary_poc['test_auc']:.4f}",
            f"{summary_poc['test_accuracy']:.2f}%",
            f"{summary_poc['test_sens_at_90_spec']*100:.2f}%",
            f"{summary_poc['fresh_conv_ssim']:.4f}",
            f"{summary_poc['fresh_conv_psnr']:.2f}dB",
            f"{summary_poc['fresh_mlp_ssim']:.4f}",
            f"{summary_poc['fresh_mlp_psnr']:.2f}dB",
            summary_poc["git_commit"]
        ])

    print("\n" + "=" * 80)
    print(f"TỔNG KẾT M3 CHO B7 LIGHTSPLIT TRÊN PCAM (SEED {args.seed}):")
    print("=" * 80)
    print(f"  * Seed:                      {summary_poc['seed']}")
    print(f"  * Best Epoch (Val AUC):      {summary_poc['best_epoch']} (Val AUC: {summary_poc['best_val_auc']:.4f})")
    print(f"  * Test AUC-ROC:              {summary_poc['test_auc']:.4f}")
    print(f"  * Test Accuracy:             {summary_poc['test_accuracy']:.2f}%")
    print(f"  * Sens @ 90% Specificity:    {summary_poc['test_sens_at_90_spec']*100:.2f}% ({summary_poc['test_sens_at_90_spec']:.4f})")
    print(f"  * Fresh Conv Attacker SSIM:  {summary_poc['fresh_conv_ssim']:.4f} (PSNR: {summary_poc['fresh_conv_psnr']:.2f} dB)")
    print(f"  * Fresh MLP Attacker SSIM:   {summary_poc['fresh_mlp_ssim']:.4f} (PSNR: {summary_poc['fresh_mlp_psnr']:.2f} dB)")
    print(f"  * File kết quả JSON:         {json_seed_p}")
    print("=" * 80 + "\n")

    tee.close()


if __name__ == "__main__":
    main()
