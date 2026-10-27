#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pcam_b0.py
==============================================================================
BƯỚC 2: ĐẶC TẢ MÔ HÌNH VÀ HUẤN LUYỆN BASELINE B0 TRÊN PATCHCAMELYON (PCAM)
==============================================================================
Đặc tả kỹ thuật:
1. Kiến trúc:
   - ResNet-18 từ CIFAR với 2 thay đổi theo yêu cầu:
     * Đầu vào: 64x64x3.
     * Stem có stride 2: conv1 3x3 stride 2 -> đầu ra client (stem + layer1) có shape:
       (B, 64, 32, 32), tổng số chiều D = 65,536 (đồng nhất với CIFAR!).
     * Server giữ nguyên: nhận z (B, 64, 32, 32) -> số lớp ra: 2 (binary).
   - Chốt chặn đầu script: assert shape đầu ra client là (64, 32, 32) và D = 65,536.
2. Huấn luyện:
   - SGD lr 0.1, momentum 0.9, weight decay 5e-4.
   - 100 Epochs, MultiStepLR tại mốc [50, 75] (gamma=0.1).
   - Batch size 128 (giống CIFAR).
3. Augmentation:
   - Lật ngang (RandomHorizontalFlip), lật dọc (RandomVerticalFlip).
   - Xoay ngẫu nhiên bội số 90 độ (RandomRotate90).
   - Bỏ RandomCrop có padding (do mô học không có hướng cố định).
4. Chuẩn hóa:
   - Sử dụng mean/std được tính từ Bước 1 (lưu trong pcam_stats.json).
5. Quản lý Checkpoints:
   - Lưu checkpoint sau mỗi epoch (last_checkpoint.pt).
   - Giữ riêng checkpoint có AUC Validation cao nhất (best_auc_b0_client.pt, best_auc_b0_server.pt).
6. Resume:
   - Cờ --resume: Tự động tiếp tục từ epoch cuối nếu Google Colab ngắt kết nối.
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
from torch.optim.lr_scheduler import MultiStepLR

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel, ServerModel
from src.data.medical import get_pcam_dataloaders
from src.metrics.clinical import compute_binary_clinical_metrics


class TeeLogger:
    """Tự động nhân đôi mọi luồng in (stdout/stderr) ra file log trên đĩa Drive."""
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
    """Lấy mã commit Git hiện tại để đảm bảo tính tái lập (reproducibility)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "fd4ebd52c8dd6698dd722fd6e3e9554249e9a9ab"


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện B0 Vanilla Split Learning trên PCAM")
    parser.add_argument("--data-dir", type=str, default="./data/pcam_processed", help="Thư mục chứa mảng .npy đã tiền xử lý")
    parser.add_argument("--output-dir", type=str, default="./output/pcam_b0_vanilla", help="Thư mục lưu kết quả và checkpoint")
    parser.add_argument("--log-file", type=str, default=None, help="Đường dẫn file ghi log (mặc định: {output_dir}/train_seed_{seed}.log)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate SGD (mặc định: 0.1)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum SGD (mặc định: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay (mặc định: 5e-4)")
    parser.add_argument("--milestones", nargs="+", type=int, default=[50, 75], help="Milestones cho MultiStepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Tỷ lệ giảm LR tại milestones")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên cố định")
    parser.add_argument("--num-workers", type=int, default=None, help="Số worker nạp dữ liệu")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint cuối nếu bị ngắt")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm nhanh 1 epoch trên vài batch")
    return parser.parse_args()


def evaluate_split(client, server, data_loader, criterion, device, max_batches=None):
    """Đánh giá loss, acc và các chỉ số lâm sàng (AUC-ROC, Sensitivity, Specificity)."""
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
        # Chỉ tính AUC nếu có cả 2 lớp trong tập mẫu
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

    # Thiết lập TeeLogger để tự động ghi log ra file trên Google Drive
    log_file_path = args.log_file or os.path.join(args.output_dir, f"train_pcam_b0_seed_{args.seed}.log")
    tee_logger = TeeLogger(log_file_path)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    git_commit_hash = get_git_commit()

    print("=" * 80)
    print("BƯỚC 2 & 3: BẮT ĐẦU HUẤN LUYỆN SPLIT LEARNING B0 (VANILLA) TRÊN PCAM")
    print(f"Thiết bị: {device} | Seed: {args.seed} | Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Milestones: {args.milestones} (gamma={args.gamma})")
    print(f"Git Commit: {git_commit_hash[:7]} ({git_commit_hash})")
    print(f"Thư mục dữ liệu: {args.data_dir}")
    print(f"Thư mục lưu kết quả: {args.output_dir}")
    print(f"File ghi log trực tiếp: {log_file_path}")
    print("=" * 80)

    # 1. Khởi tạo Mô hình: Client (stem_stride=2 cho ảnh 64x64) và Server (2 lớp)
    client = ClientModel(in_channels=3, stem_stride=2).to(device)
    server = ServerModel(num_classes=2).to(device)

    # =========================================================================
    # CHỐT CHẶN KIẾN TRÚC ĐẦU SCRIPT (MANDATORY ARCHITECTURE ASSERTION)
    # =========================================================================
    with torch.no_grad():
        test_x = torch.zeros(2, 3, 64, 64, device=device)
        test_z = client(test_x)
        assert test_z.shape == (2, 64, 32, 32), (
            f"[CHỐT CHẶN THẤT BẠI] Đầu ra client có shape {test_z.shape}, kỳ vọng (2, 64, 32, 32)!"
        )
        test_d = test_z.view(test_z.size(0), -1).shape[1]
        assert test_d == 65536, (
            f"[CHỐT CHẶN THẤT BẠI] Số chiều đặc trưng D = {test_d}, kỳ vọng 65,536!"
        )
        test_logits = server(test_z)
        assert test_logits.shape == (2, 2), (
            f"[CHỐT CHẶN THẤT BẠI] Đầu ra server logits có shape {test_logits.shape}, kỳ vọng (2, 2)!"
        )
    print("\n>>> [CHỐT CHẶN THÀNH CÔNG] Client output: (64, 32, 32), D = 65,536 | Server logits: (2) <<<\n")

    # 2. Nạp Dữ Liệu
    stats_p = os.path.join(args.data_dir, "pcam_stats.json")
    stats_dict = None
    if os.path.isfile(stats_p):
        with open(stats_p, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)
        print(f"[CHUẨN HÓA] Sử dụng Mean = {stats_dict['mean']}, Std = {stats_dict['std']} từ {stats_p}")

    train_loader, val_loader, test_loader, meta = get_pcam_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stats=stats_dict
    )
    print(f"[DỮ LIỆU] Train: {meta['num_train']:,} | Val: {meta['num_val']:,} | Test: {meta['num_test']:,}")

    # 3. Optimizers & Schedulers
    trainable_params = list(client.parameters()) + list(server.parameters())
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    # 4. Quản lý Resume
    start_epoch = 1
    best_val_auc = -1.0
    best_epoch = 1
    history = []
    last_ckpt_path = os.path.join(ckpt_dir, "last_checkpoint.pt")
    best_all_p = os.path.join(ckpt_dir, "best_checkpoint.pt")

    if os.path.isfile(best_all_p):
        try:
            ckpt_b = torch.load(best_all_p, map_location=device, weights_only=False)
            best_epoch = ckpt_b.get("epoch", 1)
            best_val_auc = ckpt_b.get("val_auc", -1.0)
        except Exception:
            pass

    if args.resume and os.path.isfile(last_ckpt_path):
        print(f"[RESUME] Đang khôi phục từ checkpoint: {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        start_epoch = ckpt["epoch"] + 1
        client.load_state_dict(ckpt["client_state_dict"])
        server.load_state_dict(ckpt["server_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_auc = max(best_val_auc, ckpt.get("best_val_auc", -1.0))
        history = ckpt.get("history", [])
        print(f"[RESUME] Khôi phục thành công! Tiếp tục từ Epoch {start_epoch} (Best Val AUC trước đó: {best_val_auc:.4f} tại Epoch {best_epoch})")

    # 5. Vòng Lặp Huấn Luyện
    total_epochs = 1 if args.dry_run else args.epochs
    max_b = 2 if args.dry_run else None

    print("\nBẮT ĐẦU TIẾN TRÌNH HUẤN LUYỆN:")
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

            # Single Forward Pass qua Split Boundary
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

        # Đánh giá trên tập Validation
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

        # Kiểm tra điều kiện Bước 3 tại Epoch 1:
        if epoch == 1:
            if val_auc <= 0.52:
                print("\n" + "!" * 80)
                print(f"[CẢNH BÁO BƯỚC 3] Val AUC sau Epoch 1 là {val_auc:.4f} (<= 0.52)!")
                print("AUC quá gần 0.50 có thể là dấu hiệu lệch nhãn hoặc tiền xử lý dữ liệu bị lỗi.")
                print("Khuyến nghị: Dừng huấn luyện và kiểm tra lại phân phối nhãn / dữ liệu.")
                print("!" * 80 + "\n", flush=True)
            else:
                print(f"\n>>> [KIỂM TRA BƯỚC 3 ĐẠT] Val AUC sau Epoch 1 là {val_auc:.4f} (> 0.52). Tín hiệu học tốt, loss đang giảm đều! <<<\n", flush=True)

        record = {
            "epoch": epoch,
            "lr": lr_curr,
            "train_loss": train_loss_mean,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_clinical": val_clin
        }
        history.append(record)

        # 6. Lưu checkpoint sau mỗi epoch (last_checkpoint.pt)
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
        }, last_ckpt_path)

        # 7. Giữ riêng bản checkpoint có AUC Val cao nhất
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_client_p = os.path.join(ckpt_dir, "best_auc_b0_client.pt")
            best_server_p = os.path.join(ckpt_dir, "best_auc_b0_server.pt")

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
            }, best_all_p)
            print(f"      >>> [BEST AUC] Đã lưu mô hình tối ưu mới tại Epoch {epoch}: Val AUC = {val_auc:.4f} <<<")

    total_time = time.time() - t_start
    print("-" * 100)
    print(f"HUẤN LUYỆN HOÀN TẤT TRONG {total_time/60:.1f} PHÚT ({total_time:.1f}s)!")
    print(f"Tối ưu tại Epoch {best_epoch} với Best Validation AUC: {best_val_auc:.4f}")

    # 8. Đánh giá cuối cùng trên toàn bộ tập Test (32,768 ảnh) với checkpoint tối ưu chọn bởi Val AUC
    print("\n" + "=" * 80)
    print("ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST TOÀN PHẦN (32,768 ẢNH)...")
    print(f"Sử dụng Checkpoint tốt nhất tại Epoch {best_epoch} (Val AUC: {best_val_auc:.4f})")
    print("=" * 80)

    if os.path.isfile(best_all_p):
        ckpt_best = torch.load(best_all_p, map_location=device, weights_only=False)
        client.load_state_dict(ckpt_best["client_state_dict"])
        server.load_state_dict(ckpt_best["server_state_dict"])
        print(f"[TEST EVAL] Đã nạp thành công trọng số tối ưu từ Epoch {ckpt_best.get('epoch', best_epoch)}.")

    test_loss, test_acc, test_auc, test_clin = evaluate_split(client, server, test_loader, criterion, device, max_batches=max_b)
    test_sens_90 = float(test_clin.get("sensitivity_at_90_specificity", 0.0))

    print(f"\nKẾT QUẢ ĐỐI CHUẨN TEST B0 VANILLA TRÊN PCAM (SEED {args.seed}):")
    print(f"  - Best Epoch (theo Val AUC): {best_epoch}")
    print(f"  - Best Val AUC-ROC:         {best_val_auc:.4f}")
    print(f"  - Test Loss:                 {test_loss:.4f}")
    print(f"  - Test Accuracy:             {test_acc:.2f}%")
    print(f"  - Test AUC-ROC:              {test_auc:.4f}")
    if test_clin:
        print(f"  - Sensitivity @ 0.5:         {test_clin.get('sensitivity_0.5', 0.0)*100:.2f}%")
        print(f"  - Specificity @ 0.5:         {test_clin.get('specificity_0.5', 0.0)*100:.2f}%")
        print(f"  - Sens @ 90% Specificity:    {test_sens_90*100:.2f}% ({test_sens_90:.4f})")
        print(f"  - Youden Optimal Thresh:     {test_clin.get('youden_threshold', 0.5):.4f}")

    # Chuẩn bị cấu trúc báo cáo 4 chỉ số cốt lõi cho PoC
    summary_poc = {
        "seed": args.seed,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(round(best_val_auc, 4)),
        "test_auc": float(round(test_auc, 4)),
        "test_accuracy": float(round(test_acc, 2)),
        "test_sens_at_90_spec": float(round(test_sens_90, 4)),
        "train_time_seconds": float(round(total_time, 1)),
        "train_time_formatted": f"{int(total_time // 60)}m {int(total_time % 60):02d}s",
        "git_commit": git_commit_hash[:7],
        "git_commit_full": git_commit_hash
    }

    # 9. Lưu kết quả ra file JSON và CSV
    final_results = {
        "dataset": "pcam",
        "method": "B0_Vanilla",
        "seed": args.seed,
        "git_commit": git_commit_hash,
        "train_time_seconds": total_time,
        "train_time_formatted": f"{int(total_time // 60)}m {int(total_time % 60):02d}s",
        "input_shape": [64, 64, 3],
        "smashed_shape": [64, 32, 32],
        "D": 65536,
        "epochs": args.epochs,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_auc_roc": float(test_auc),
        "test_sens_at_90_spec": float(test_sens_90),
        "summary_for_poc": summary_poc,
        "clinical_metrics": test_clin,
        "history": history
    }

    # Lưu 2 file JSON: theo seed riêng biệt và kết quả tổng
    json_seed_p = os.path.join(args.output_dir, f"results_pcam_b0_seed_{args.seed}.json")
    with open(json_seed_p, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    json_res_p = os.path.join(args.output_dir, "results_pcam_b0.json")
    with open(json_res_p, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n[XUẤT KẾT QUẢ] Đã lưu kết quả JSON Seed {args.seed} tại: {json_seed_p}")

    csv_res_p = os.path.join(args.output_dir, f"results_pcam_b0_seed_{args.seed}.csv")
    with open(csv_res_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Seed", "Best_Epoch", "Val_AUC", "Test_AUC", "Test_Acc", "Sens@90%Spec", "Train_Time", "Git_Commit"])
        writer.writerow([
            args.seed,
            best_epoch,
            f"{best_val_auc:.4f}",
            f"{test_auc:.4f}",
            f"{test_acc:.2f}%",
            f"{test_sens_90*100:.2f}%",
            summary_poc["train_time_formatted"],
            git_commit_hash[:7]
        ])
    print(f"[XUẤT KẾT QUẢ] Đã lưu kết quả CSV tại:  {csv_res_p}")

    # In banner 4 chỉ số theo quy định Bước 4:
    print("\n" + "=" * 80)
    print(f"TỔNG KẾT BƯỚC 4 CHO SEED {args.seed} (SẴN SÀNG CHO BÁO CÁO & BẢNG POC):")
    print("=" * 80)
    print(f"  * Seed:                      {summary_poc['seed']}")
    print(f"  * Best Epoch (theo Val AUC): {summary_poc['best_epoch']}")
    print(f"  * Best Val AUC:              {summary_poc['best_val_auc']:.4f}")
    print(f"  * Test AUC-ROC:              {summary_poc['test_auc']:.4f}")
    print(f"  * Test Accuracy:             {summary_poc['test_accuracy']:.2f}%")
    print(f"  * Sens @ 90% Specificity:    {summary_poc['test_sens_at_90_spec']*100:.2f}% ({summary_poc['test_sens_at_90_spec']:.4f})")
    print(f"  * Thời gian huấn luyện:      {summary_poc['train_time_formatted']} ({summary_poc['train_time_seconds']:.1f}s)")
    print(f"  * Commit Git:                {summary_poc['git_commit']}")
    print("=" * 80)

    # 10. Vẽ đường cong tiến trình huấn luyện
    if len(history) > 1:
        plot_p = os.path.join(args.output_dir, f"curve_pcam_b0_seed_{args.seed}.png")
        eps = [h["epoch"] for h in history]
        tr_accs = [h["train_acc"] for h in history]
        va_accs = [h["val_acc"] for h in history]
        va_aucs = [h["val_auc"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(eps, tr_accs, "b-", label="Train Acc (%)", lw=2)
        ax1.plot(eps, va_accs, "g--", label="Val Acc (%)", lw=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Độ chính xác (%)")
        ax1.set_title(f"Accuracy PCAM 64x64 (B0, Seed {args.seed})")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(eps, va_aucs, "r-", label="Val AUC-ROC", lw=2)
        ax2.axhline(best_val_auc, color="gray", linestyle=":", label=f"Best AUC: {best_val_auc:.4f} (Ep {best_epoch})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC-ROC")
        ax2.set_title(f"AUC-ROC Validation PCAM (B0, Seed {args.seed})")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(plot_p, dpi=200)
        plt.close()
        print(f"[ĐỒ THỊ] Đã lưu biểu đồ huấn luyện tại: {plot_p}")

    print("\n" + "=" * 80)
    print(f"HOÀN TẤT HUẤN LUYỆN SEED {args.seed}!")
    print("=" * 80)

    tee_logger.close()


if __name__ == "__main__":
    main()
