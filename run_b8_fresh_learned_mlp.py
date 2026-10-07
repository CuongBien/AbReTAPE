#!/usr/bin/env python3
# ==============================================================================
# ĐO ĐẠC FRESH ATTACKER MẠNH NHẤT: LEARNED ADAPTIVE MLP TRÊN CLIENT B8 ĐÃ ĐÓNG BĂNG
# ==============================================================================
# Giao thức Bước 3:
# - Attacker: LearnedAdaptiveDecoderB8 (MLP Unprojector 65536 -> 512 -> 65536 + Conv Decoder)
# - Dữ liệu train: Full 50.000 ảnh CIFAR-10, 30 epochs, Adam lr=1e-3, CosineAnnealingLR
# - Client B8: Đóng băng hoàn toàn (requires_grad=False), chạy ở client.eval()
# - Đánh giá: Đúng 2.000 ảnh test cố định (seed=42) mà adversary B8 đã dùng, cùng toàn bộ 10.000 test
# - Đo đạc độ lệch G = SSIM(Fresh Learned MLP) - SSIM(In-training Adversary)
# ==============================================================================
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel
from src.data.cifar import CIFAR10_MEAN, CIFAR10_STD
from src.attacks import Decoder
from src.metrics import psnr_ssim, get_lpips_fn, calculate_lpips
from src.utils.visualize import save_reconstruction_grid


class LearnedAdaptiveDecoderB8(nn.Module):
    """
    Learned Adaptive MLP Attacker (Unprojector 65536 -> 512 -> 65536 + ConvDecoder).
    """
    def __init__(self, in_features=65536, D=65536, hidden_dim=512):
        super().__init__()
        self.unprojector = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D),
            nn.BatchNorm1d(D)
        )
        self.conv_decoder = Decoder(in_channels=64, out_channels=3)

    def forward(self, z):
        if z.dim() > 2:
            z_flat = z.view(z.size(0), -1)
        else:
            z_flat = z
        z_unproj = self.unprojector(z_flat).view(-1, 64, 32, 32)
        return self.conv_decoder(z_unproj)


def get_data_loaders(data_dir="./data", batch_size=128, test_eval_size=2000, num_workers=0, seed=42):
    mean = CIFAR10_MEAN
    std = CIFAR10_STD

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    trainset = CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    testset_eval = CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)
    testset_full = CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)

    # 2.000 ảnh test cố định đồng nhất với B8 in-training evaluation
    generator = torch.Generator().manual_seed(seed)
    test_perm = torch.randperm(len(testset_eval), generator=generator).tolist()
    test_eval_subset = Subset(testset_eval, test_perm[:test_eval_size])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_eval_loader = DataLoader(test_eval_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_full_loader = DataLoader(testset_full, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, test_eval_loader, test_full_loader


@torch.no_grad()
def evaluate_fresh_attacker(client, attacker, loader, device, lpips_fn=None):
    """
    Đánh giá Fresh Attacker trên loader đã cho với client.eval().
    """
    client.eval()
    attacker.eval()

    total_samples = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    mse_norm_sum = 0.0

    criterion = nn.MSELoss()

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        b = x.size(0)
        z = client(x)
        x_rec = attacker(z)

        mse_norm = criterion(x_rec, x).item()
        p, s = psnr_ssim(x, x_rec, CIFAR10_MEAN, CIFAR10_STD)
        lp = calculate_lpips(x, x_rec, CIFAR10_MEAN, CIFAR10_STD, lpips_fn) if lpips_fn else None

        mse_norm_sum += mse_norm * b
        psnr_sum += p * b
        ssim_sum += s * b
        if lp is not None:
            lpips_sum += lp * b
        total_samples += b

    n = max(total_samples, 1)
    res = {
        "mse_norm": mse_norm_sum / n,
        "psnr": psnr_sum / n,
        "ssim": ssim_sum / n,
        "lpips": (lpips_sum / n) if lpips_fn else None
    }
    return res


def train_fresh_mlp_for_lambda(lam, client_path, args, train_loader, test_eval_loader, test_full_loader, device, lpips_fn):
    print("=" * 80)
    print(f"HUẤN LUYỆN FRESH LEARNED MLP CHO B8 CLIENT (λ = {lam}, Seed = {args.seed})")
    print(f"Checkpoint Client: {client_path}")
    print("=" * 80, flush=True)

    if not os.path.isfile(client_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {client_path}")

    # 1. Nạp Client B8 và đóng băng hoàn toàn
    client = ClientModel().to(device)
    ckpt = torch.load(client_path, map_location=device)
    if isinstance(ckpt, dict) and "client" in ckpt:
        client.load_state_dict(ckpt["client"])
    else:
        client.load_state_dict(ckpt)
    client.eval()
    for p in client.parameters():
        p.requires_grad = False

    # 2. Khởi tạo Fresh Attacker
    attacker = LearnedAdaptiveDecoderB8(in_features=65536, D=65536, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_ssim = 0.0
    best_psnr = 0.0
    best_ep = 0

    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        attacker.train()
        train_loss_sum = 0.0
        n_train = 0

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            b = x.size(0)

            with torch.no_grad():
                z = client(x)

            optimizer.zero_grad()
            x_rec = attacker(z)
            loss = criterion(x_rec, x)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * b
            n_train += b

        scheduler.step()
        train_mse = train_loss_sum / max(n_train, 1)

        # Đánh giá trên 2.000 ảnh test cố định
        if ep % args.eval_freq == 0 or ep == args.epochs:
            eval_res = evaluate_fresh_attacker(client, attacker, test_eval_loader, device, lpips_fn=None)
            cur_psnr = eval_res["psnr"]
            cur_ssim = eval_res["ssim"]

            if cur_ssim > best_ssim:
                best_ssim = cur_ssim
                best_psnr = cur_psnr
                best_ep = ep
                # Lưu checkpoint tốt nhất
                save_path = os.path.join(args.output_dir, "checkpoints", f"b8_fresh_mlp_lam{lam}_s{args.seed}_best.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "attacker": attacker.state_dict(),
                    "epoch": ep,
                    "lambda": lam,
                    "ssim": best_ssim,
                    "psnr": best_psnr
                }, save_path)

            print(f"  [Epoch {ep:2d}/{args.epochs}] Train MSE: {train_mse:.4f} | Test(2k) PSNR: {cur_psnr:.2f} dB | SSIM: {cur_ssim:.4f} (Best: {best_ssim:.4f} @ Ep {best_ep})", flush=True)

    train_time = time.time() - t_start
    print(f"[HOÀN TẤT] Huấn luyện λ={lam} xong trong {train_time:.1f}s. Đang nạp model tốt nhất để đánh giá toàn diện...")

    # Nạp best checkpoint để đánh giá toàn diện
    best_ckpt_path = os.path.join(args.output_dir, "checkpoints", f"b8_fresh_mlp_lam{lam}_s{args.seed}_best.pt")
    if os.path.isfile(best_ckpt_path):
        b_ckpt = torch.load(best_ckpt_path, map_location=device)
        attacker.load_state_dict(b_ckpt["attacker"])

    # Đánh giá trên 2.000 test cố định (kèm LPIPS)
    res_2k = evaluate_fresh_attacker(client, attacker, test_eval_loader, device, lpips_fn=lpips_fn)
    # Đánh giá trên 10.000 test đầy đủ
    res_10k = evaluate_fresh_attacker(client, attacker, test_full_loader, device, lpips_fn=None)

    # In-training adversary baseline SSIM từ kết quả trước
    in_train_ssim_map = {
        0.1: 0.3938,
        0.5: 0.2979,
        1.0: 0.2776
    }
    in_train_ssim = in_train_ssim_map.get(lam, None)
    gap_g = (res_2k["ssim"] - in_train_ssim) if in_train_ssim is not None else None

    print(f"\n[KẾT QUẢ λ={lam}]")
    print(f"  - In-training Adv SSIM: {in_train_ssim:.4f}" if in_train_ssim else "  - In-training Adv SSIM: N/A")
    print(f"  - Fresh MLP (2k test): SSIM = {res_2k['ssim']:.4f} | PSNR = {res_2k['psnr']:.2f} dB | LPIPS = {res_2k['lpips']:.4f}")
    print(f"  - Fresh MLP (10k test): SSIM = {res_10k['ssim']:.4f} | PSNR = {res_10k['psnr']:.2f} dB")
    if gap_g is not None:
        print(f"  -> Discrepancy G = SSIM(MLP) - SSIM(Adv) = {gap_g:+.4f}")
    print("-" * 80, flush=True)

    # Lưu ảnh trực quan hóa
    grid_path = os.path.join(args.output_dir, f"rec_fresh_mlp_lam{lam}_s{args.seed}.png")
    save_reconstruction_grid(client, attacker, test_eval_loader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=grid_path, num_images=8)

    return {
        "lambda": lam,
        "seed": args.seed,
        "in_train_adv_ssim": in_train_ssim,
        "fresh_mlp_ssim_2k": round(res_2k["ssim"], 4),
        "fresh_mlp_psnr_2k": round(res_2k["psnr"], 2),
        "fresh_mlp_lpips_2k": round(res_2k["lpips"], 4) if res_2k["lpips"] else None,
        "fresh_mlp_mse_norm_2k": round(res_2k["mse_norm"], 4),
        "fresh_mlp_ssim_10k": round(res_10k["ssim"], 4),
        "fresh_mlp_psnr_10k": round(res_10k["psnr"], 2),
        "discrepancy_G": round(gap_g, 4) if gap_g is not None else None,
        "best_epoch": best_ep,
        "training_time_s": round(train_time, 1)
    }


def main():
    parser = argparse.ArgumentParser(description="Fresh Learned Adaptive MLP Attacker trên B8 đã đóng băng")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.5, 1.0], help="Danh sách lambda cần chạy")
    parser.add_argument("--seed", type=int, default=42, help="Seed của checkpoint và data split (mặc định: 42)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện decoder (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam (mặc định: 1e-3)")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Chiều ẩn MLP (mặc định: 512)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá mỗi N epochs (mặc định: 5)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử 1 epoch rút gọn để kiểm tra lỗi cú pháp/logic")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 1
        args.eval_freq = 1
        args.lambdas = [0.1]
        args.output_dir = os.path.join(args.output_dir, "dry_run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    print("=" * 85)
    print("BƯỚC 3: ĐO ĐẠC FRESH LEARNED ADAPTIVE MLP TRÊN CLIENT B8 ĐÃ ĐÓNG BĂNG")
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Danh sách λ: {args.lambdas} | Seed: {args.seed}")
    print(f"Kiến trúc Attacker: LearnedAdaptiveDecoderB8 (Unprojector 65536 -> {args.hidden_dim} -> 65536 + ConvDecoder)")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print("=" * 85, flush=True)

    train_loader, test_eval_loader, test_full_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_eval_size=2000,
        num_workers=args.num_workers,
        seed=args.seed
    )

    lpips_fn = get_lpips_fn(device=device)

    all_results = []
    for lam in args.lambdas:
        client_ckpt_name = f"b8_conv_lam{lam}_s{args.seed}_client.pt"
        client_ckpt_path = os.path.join(args.output_dir, "checkpoints", client_ckpt_name)

        if not os.path.isfile(client_ckpt_path):
            print(f"[WARN] Bỏ qua λ={lam}: Không tìm thấy {client_ckpt_path}")
            continue

        res = train_fresh_mlp_for_lambda(
            lam=lam,
            client_path=client_ckpt_path,
            args=args,
            train_loader=train_loader,
            test_eval_loader=test_eval_loader,
            test_full_loader=test_full_loader,
            device=device,
            lpips_fn=lpips_fn
        )
        all_results.append(res)

    # Lưu kết quả tổng hợp
    json_path = os.path.join(args.output_dir, "results_b8_fresh_learned_mlp.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output_dir, "results_b8_fresh_learned_mlp.csv")
    if all_results:
        import csv
        keys = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

    print("\n" + "=" * 85)
    print("TỔNG HỢP KẾT QUẢ FRESH LEARNED MLP TRÊN B8:")
    print(f"{'Lambda':<8} | {'In-train SSIM':<14} | {'Fresh MLP (2k)':<15} | {'PSNR (2k)':<10} | {'LPIPS (2k)':<10} | {'Discrepancy G':<14}")
    print("-" * 85)
    for r in all_results:
        in_s = f"{r['in_train_adv_ssim']:.4f}" if r['in_train_adv_ssim'] is not None else "N/A"
        g_s = f"{r['discrepancy_G']:+.4f}" if r['discrepancy_G'] is not None else "N/A"
        lp_s = f"{r['fresh_mlp_lpips_2k']:.4f}" if r['fresh_mlp_lpips_2k'] is not None else "N/A"
        print(f"{r['lambda']:<8} | {in_s:<14} | {r['fresh_mlp_ssim_2k']:<15.4f} | {r['fresh_mlp_psnr_2k']:<10.2f} | {lp_s:<10} | {g_s:<14}")
    print("=" * 85)
    print(f"Đã lưu kết quả tại: {json_path} và {csv_path}")


if __name__ == "__main__":
    main()
