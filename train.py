#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
==============================================================================
UNIFIED TRAINING ENTRYPOINT CHO SPLIT LEARNING & AR-TAPE
Hỗ trợ cả CIFAR-10 và Y Tế (HAM10000, PCAM), cùng các cơ chế phòng thủ:
- Vanilla B0 (không phòng thủ)
- B7 LightSplit (Fixed Orthogonal k=512, 1024, 2048)
- B8 Adversarial (Full D, in-train adversary)
- AR-TAPE v1 (Mode F k=2048, fixed bottleneck, min-max adversarial)
==============================================================================
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    ClientModel,
    ServerModel,
    ClientWithBottleneck,
    ServerWithBottleneck,
    LearnedAdaptiveDecoder
)
from src.data import get_dataloaders
from src.metrics import psnr_ssim, distance_correlation
from src.utils.visualize import save_reconstruction_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Split Learning Training Engine")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "ham10000", "pcam"], help="Dataset")
    parser.add_argument("--defense", type=str, default="ar_tape", choices=["none", "b7_lightsplit", "b8_adversarial", "ar_tape"], help="Cơ chế phòng thủ")
    parser.add_argument("--k", type=int, default=2048, help="Số chiều bottleneck (mặc định: 2048)")
    parser.add_argument("--m", type=int, default=512, help="Số chiều unprojector (mặc định: 512)")
    parser.add_argument("--D", type=int, default=65536, help="Số chiều đặc trưng cut layer (mặc định: 65536)")
    parser.add_argument("--lambda-val", type=float, default=0.25, help="Hệ số đối kháng lambda (mặc định: 0.25)")
    parser.add_argument("--cap", type=float, default=1.5586, help="Ngưỡng chặn MSE loss (mặc định: 1.5586)")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Số epochs warmup (mặc định: 10)")
    parser.add_argument("--unprojector-mode", type=str, default="linear_fixed", choices=["linear_fixed", "linear_learned", "mlp"])
    parser.add_argument("--fixed-bottleneck", action="store_true", default=True, help="Cố định ma trận chiếu R tại client")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate SGD cho Client & Server")
    parser.add_argument("--adv-lr", type=float, default=1e-3, help="Learning rate Adam cho In-training Adversary")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "multistep", "none"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="./data", help="Thư mục chứa dữ liệu")
    parser.add_argument("--output-dir", type=str, default="./output/training_run", help="Thư mục lưu kết quả")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm nhanh 1 epoch")
    return parser.parse_args()


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

    print("=" * 80)
    print(f"BẮT ĐẦU HUẤN LUYỆN SPLIT LEARNING: {args.defense.upper()}")
    print(f"Dataset: {args.dataset.upper()} | Thiết bị: {device} | Seed: {args.seed}")
    print(f"Cấu hình: k={args.k} | λ={args.lambda_val} | Cap={args.cap} | Epochs={args.epochs}")
    print(f"Thư mục lưu kết quả: {args.output_dir}")
    print("=" * 80, flush=True)

    # 1. Nạp Dữ Liệu
    train_loader, test_loader, meta = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )
    if train_loader is None or test_loader is None:
        print(f"[LỖI] Không thể nạp dataset '{args.dataset}'. Vui lòng kiểm tra dữ liệu tại {args.data_dir}.")
        sys.exit(1)

    num_classes = meta["num_classes"]
    in_channels = meta["in_channels"]

    # 2. Khởi tạo Client và Server
    is_bottleneck = args.defense in ["b7_lightsplit", "ar_tape"]
    use_fixed = args.fixed_bottleneck or (args.defense == "b7_lightsplit")
    stem_stride = 2 if args.dataset == "pcam" else 1

    if is_bottleneck:
        client = ClientWithBottleneck(
            k=args.k, D=args.D, init_ortho=True, seed=args.seed, fixed_bottleneck=use_fixed, in_channels=in_channels, stem_stride=stem_stride
        ).to(device)
        ortho_q = client.R if client.fixed_bottleneck else client.bottleneck.weight.t().detach()
        server = ServerWithBottleneck(
            k=args.k, m=args.m, D=args.D, num_classes=num_classes, unprojector_mode=args.unprojector_mode, ortho_matrix=ortho_q, seed=args.seed
        ).to(device)
    else:
        client = ClientModel(in_channels=in_channels, stem_stride=stem_stride).to(device)
        server = ServerModel(num_classes=num_classes).to(device)

    # Chốt chặn kiến trúc: kiểm tra shape đầu ra client và D = 65,536
    with torch.no_grad():
        img_h, img_w = meta.get("img_size", (32, 32))
        dummy_in = torch.zeros(2, in_channels, img_h, img_w, device=device)
        dummy_z = client(dummy_in)
        if not is_bottleneck:
            assert dummy_z.shape == (2, 64, 32, 32), (
                f"[CHỐT CHẶN THẤT BẠI] Đầu ra client có shape {dummy_z.shape}, kỳ vọng (2, 64, 32, 32)!"
            )
            assert dummy_z.view(2, -1).shape[1] == 65536, (
                f"[CHỐT CHẶN THẤT BẠI] Smashed data D = {dummy_z.view(2, -1).shape[1]}, kỳ vọng 65536!"
            )
        else:
            assert dummy_z.shape == (2, args.k), (
                f"[CHỐT CHẶN THẤT BẠI] Đầu ra bottleneck có shape {dummy_z.shape}, kỳ vọng (2, {args.k})!"
            )
        dummy_logits = server(dummy_z)
        assert dummy_logits.shape == (2, num_classes), (
            f"[CHỐT CHẶN THẤT BẠI] Đầu ra logits có shape {dummy_logits.shape}, kỳ vọng (2, {num_classes})!"
        )
    print(f"[CHỐT CHẶN KIẾN TRÚC ĐẠT] Input ({in_channels}, {img_h}, {img_w}) -> Client -> Smashed data D={args.k if is_bottleneck else 65536} -> Server -> {num_classes} classes.")

    # 3. Khởi tạo In-training Adversary nếu dùng cơ chế đối kháng
    use_adversary = args.defense in ["b8_adversarial", "ar_tape"] and args.lambda_val > 0
    if use_adversary:
        adv = LearnedAdaptiveDecoder(k=args.k if is_bottleneck else args.D, D=args.D, hidden_dim=args.m).to(device)
        opt_adv = torch.optim.Adam(adv.parameters(), lr=args.adv_lr)
    else:
        adv = None
        opt_adv = None

    # 4. Optimizers & Schedulers
    trainable_cs = [p for p in client.parameters() if p.requires_grad] + [p for p in server.parameters() if p.requires_grad]
    opt_cs = torch.optim.SGD(trainable_cs, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.scheduler == "cosine":
        sched_cs = CosineAnnealingLR(opt_cs, T_max=args.epochs)
    elif args.scheduler == "multistep":
        sched_cs = MultiStepLR(opt_cs, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1)
    else:
        sched_cs = None

    criterion_task = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    total_epochs = 1 if args.dry_run else args.epochs
    history = []
    t0 = time.time()

    for ep in range(1, total_epochs + 1):
        client.train()
        server.train()
        if adv:
            adv.train()

        total_loss, correct, total = 0.0, 0, 0
        total_rec_loss = 0.0

        for b_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Single Forward Pass qua Client
            z = client(x)

            # Max Step (Cập nhật In-training Adversary)
            if adv is not None:
                z_det = z.detach()
                x_rec = adv(z_det)
                loss_adv_train = criterion_mse(x_rec, x)

                opt_adv.zero_grad()
                loss_adv_train.backward()
                opt_adv.step()

            # Min Step (Cập nhật Client + Server)
            out = server(z)
            loss_task = criterion_task(out, y)

            if adv is not None:
                adv.eval()
                for p in adv.parameters():
                    p.requires_grad = False

                x_rec_min = adv(z)
                rec_mse = criterion_mse(x_rec_min, x)
                rec_capped = torch.clamp(rec_mse, max=args.cap)

                # Warmup lambda tuyến tính
                curr_lam = args.lambda_val * min(1.0, ep / max(1, args.warmup_epochs))
                loss_cs = loss_task - curr_lam * rec_capped
                total_rec_loss += rec_mse.item()

                for p in adv.parameters():
                    p.requires_grad = True
                adv.train()
            else:
                loss_cs = loss_task

            opt_cs.zero_grad()
            loss_cs.backward()
            opt_cs.step()

            total_loss += loss_task.item() * x.size(0)
            preds = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            if args.dry_run and b_idx >= 2:
                break

        if sched_cs:
            sched_cs.step()

        train_acc = correct / max(1, total) * 100.0
        train_loss = total_loss / max(1, total)

        # Đánh giá cuối epoch
        client.eval()
        server.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for b_idx, (tx, ty) in enumerate(test_loader):
                tx, ty = tx.to(device), ty.to(device)
                tz = client(tx)
                tout = server(tz)
                test_correct += (tout.argmax(dim=-1) == ty).sum().item()
                test_total += tx.size(0)
                if args.dry_run and b_idx >= 2:
                    break

        test_acc = test_correct / max(1, test_total) * 100.0

        # Đo dCor định kỳ (mỗi 10 epoch hoặc epoch cuối)
        dcor_val = None
        if ep % 10 == 0 or ep == total_epochs:
            with torch.no_grad():
                sub_x = x[:min(128, x.size(0))]
                sub_z = client(sub_x)
                dcor_val = float(distance_correlation(sub_x, sub_z))

        rec = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "dcor": dcor_val
        }
        history.append(rec)

        dcor_str = f" | dCor: {dcor_val:.4f}" if dcor_val is not None else ""
        print(f"  [Ep {ep:3d}/{total_epochs:3d}] Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%{dcor_str}", flush=True)

    dt = time.time() - t0

    # 5. Lưu Checkpoints
    prefix = f"{args.dataset}_{args.defense}_k{args.k}_lam{args.lambda_val}_s{args.seed}"
    client_ckpt = os.path.join(ckpt_dir, f"{prefix}_client.pt")
    server_ckpt = os.path.join(ckpt_dir, f"{prefix}_server.pt")
    torch.save(client.state_dict(), client_ckpt)
    torch.save(server.state_dict(), server_ckpt)

    # 6. Lưu Kết Quả JSON
    summary = {
        prefix: {
            "dataset": args.dataset,
            "defense": args.defense,
            "k": args.k,
            "lambda": args.lambda_val,
            "seed": args.seed,
            "final_full_test_acc": test_acc,
            "final_dcor": dcor_val,
            "training_time_s": dt,
            "client_checkpoint": client_ckpt,
            "server_checkpoint": server_ckpt,
            "history": history
        }
    }
    json_path = os.path.join(args.output_dir, "results_training.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"HOÀN TẤT HUẤN LUYỆN TRONG {dt:.1f}s")
    print(f"Acc Cuối: {test_acc:.2f}% | dCor: {dcor_val}")
    print(f"Đã lưu checkpoint: {client_ckpt}")
    print(f"Đã lưu kết quả JSON: {json_path}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
