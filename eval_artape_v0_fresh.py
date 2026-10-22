#!/usr/bin/env python3
# ==============================================================================
# FRESH ATTACKER EVALUATION CHO AR-TAPE v0 ĐÃ ĐÓNG BĂNG
# ==============================================================================
# - Testbed: CIFAR-10, Split Learning
# - Client AR-TAPE v0: Đóng băng hoàn toàn (requires_grad=False), chạy ở client.eval()
# - Hỗ trợ 2 kiến trúc Attacker:
#     * 'mlp': LearnedAdaptiveDecoder (MLP Unprojector 1024 -> 512 -> 65536 + ConvDecoder, ~34M params)
#     * 'conv': Transpose-Lift ConvDecoder (Linear lift zt @ W_bottleneck -> ConvDecoder 64->3)
# - Dữ liệu: 50.000 ảnh CIFAR-10, 30 epochs, Adam lr=1e-3, CosineAnnealingLR
# - Đánh giá: 2.000 test cố định (seed=42) và toàn bộ 10.000 test
# - Đầu ra: results_artape_v0_fresh_{attacker}.csv / .json, checkpoint best, ảnh grid
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
from train_artape_v0 import ClientWithBottleneck, LearnedAdaptiveDecoder


class TransposeLiftConvDecoder(nn.Module):
    """
    Attacker Conv cho biểu diễn nén zt in R^{B x k}:
    Dùng chính chuyển vị của ma trận bottleneck W^T để nâng zt lên không gian phẳng D=65536
    rồi reshape về (B, 64, 32, 32) và đưa qua ConvDecoder 4 tầng chuẩn.
    Tương tự cơ chế Conv lift trong B7 LightSplit.
    """
    def __init__(self, bottleneck_weight, in_channels=64, out_channels=3):
        super().__init__()
        # bottleneck_weight có shape [k, D] -> chuyển vị [D, k]
        self.register_buffer("W_lift", bottleneck_weight.t().clone())
        self.conv_decoder = Decoder(in_channels=in_channels, out_channels=out_channels)

    def forward(self, zt):
        if zt.dim() > 2:
            zt = zt.view(zt.size(0), -1)
        # zt: [B, k] @ W_lift.t() [k, D] -> [B, D]
        zh = torch.matmul(zt, self.W_lift.t())
        zh_tensor = zh.view(zt.size(0), 64, 32, 32)
        return self.conv_decoder(zh_tensor)


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
        zt = client(x)
        x_rec = attacker(zt)

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
    return {
        "mse_norm": mse_norm_sum / n,
        "psnr": psnr_sum / n,
        "ssim": ssim_sum / n,
        "lpips": (lpips_sum / n) if lpips_fn else None
    }


def train_fresh_attacker(attacker_type, client_ckpt_path, lam, args, train_loader, test_eval_loader, test_full_loader, device, lpips_fn):
    print("=" * 85)
    print(f"HUẤN LUYỆN FRESH ATTACKER [{attacker_type.upper()}] CHO AR-TAPE v0 (λ = {lam}, Seed = {args.seed})")
    print(f"Checkpoint Client: {client_ckpt_path}")
    print("=" * 85, flush=True)

    # 1. Nạp Client AR-TAPE v0 và đóng băng
    ckpt = torch.load(client_ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "client" in ckpt:
        sd = ckpt["client"]
    else:
        sd = ckpt
    is_fixed = ("R" in sd) or getattr(args, "fixed_bottleneck", False)
    client = ClientWithBottleneck(k=args.k, D=args.D, init_ortho=False, seed=args.seed, fixed_bottleneck=is_fixed).to(device)
    client.load_state_dict(sd)
    client.eval()
    for p in client.parameters():
        p.requires_grad = False

    # 2. Khởi tạo Attacker
    if attacker_type == "mlp":
        attacker = LearnedAdaptiveDecoder(k=args.k, D=args.D, hidden_dim=args.hidden_dim).to(device)
    elif attacker_type == "conv":
        bw = client.get_bottleneck_weight().data if hasattr(client, "get_bottleneck_weight") else client.bottleneck.weight.data
        attacker = TransposeLiftConvDecoder(bw).to(device)
    else:
        raise ValueError(f"Không hỗ trợ attacker_type: {attacker_type}")

    optimizer = torch.optim.Adam(attacker.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = None if getattr(args, "constant_lr", False) else CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_ssim = 0.0
    best_psnr = 0.0
    best_ep = 0

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    t_start = time.time()
    history = []
    for ep in range(1, args.epochs + 1):
        attacker.train()
        train_loss_sum = 0.0
        n_train = 0

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            b = x.size(0)

            with torch.no_grad():
                zt = client(x)

            optimizer.zero_grad()
            x_rec = attacker(zt)
            loss = criterion(x_rec, x)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * b
            n_train += b

        if scheduler is not None:
            scheduler.step()
        train_mse = train_loss_sum / max(n_train, 1)

        if ep % args.eval_freq == 0 or ep == args.epochs:
            eval_res = evaluate_fresh_attacker(client, attacker, test_eval_loader, device, lpips_fn=None)
            cur_psnr = eval_res["psnr"]
            cur_ssim = eval_res["ssim"]

            history.append({
                "epoch": ep,
                "train_mse": round(train_mse, 4),
                "test_psnr": round(cur_psnr, 2),
                "test_ssim": round(cur_ssim, 4)
            })

            if cur_ssim > best_ssim:
                best_ssim = cur_ssim
                best_psnr = cur_psnr
                best_ep = ep
                save_path = os.path.join(checkpoint_dir, f"artape_v0_fresh_{attacker_type}_lam{lam}_s{args.seed}_best.pt")
                torch.save({
                    "attacker": attacker.state_dict(),
                    "epoch": ep,
                    "lambda": lam,
                    "ssim": best_ssim,
                    "psnr": best_psnr
                }, save_path)

            print(f"  [Epoch {ep:2d}/{args.epochs}] Train MSE: {train_mse:.4f} | Test(2k) PSNR: {cur_psnr:.2f} dB | SSIM: {cur_ssim:.4f} (Best: {best_ssim:.4f} @ Ep {best_ep})", flush=True)

    train_time = time.time() - t_start
    print(f"[HOÀN TẤT] Huấn luyện {attacker_type} λ={lam} xong trong {train_time:.1f}s. Đang nạp model tốt nhất để đánh giá...")

    best_ckpt_path = os.path.join(checkpoint_dir, f"artape_v0_fresh_{attacker_type}_lam{lam}_s{args.seed}_best.pt")
    if os.path.isfile(best_ckpt_path):
        b_ckpt = torch.load(best_ckpt_path, map_location=device)
        attacker.load_state_dict(b_ckpt["attacker"])

    res_2k = evaluate_fresh_attacker(client, attacker, test_eval_loader, device, lpips_fn=lpips_fn)
    res_10k = evaluate_fresh_attacker(client, attacker, test_full_loader, device, lpips_fn=None)

    # Vẽ và lưu biểu đồ learning curve theo epoch
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eps = [h["epoch"] for h in history]
        ssims = [h["test_ssim"] for h in history]
        psnrs = [h["test_psnr"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(eps, ssims, "b-o", markersize=3, label="Test SSIM (2k)")
        ax1.axhline(0.30, color="r", linestyle="--", label="PoC Ceiling 0.30")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("SSIM")
        ax1.set_title(f"Fresh {attacker_type.upper()} SSIM (Max: {best_ssim:.4f})")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(eps, psnrs, "g-o", markersize=3, label="Test PSNR (2k)")
        ax2.axhline(15.0, color="r", linestyle="--", label="PoC Ceiling 15 dB")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("PSNR (dB)")
        ax2.set_title(f"Fresh {attacker_type.upper()} PSNR (Max: {best_psnr:.2f} dB)")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        curve_path = os.path.join(args.output_dir, f"curve_fresh_{attacker_type}_lam{lam}_s{args.seed}.png")
        plt.savefig(curve_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Không vẽ được learning curve: {e}")

    # Đọc in-train adversary SSIM từ results_artape_v0.json
    in_train_ssim = None
    v0_json = os.path.join(args.output_dir, "results_artape_v0.json")
    if os.path.isfile(v0_json):
        try:
            with open(v0_json, "r", encoding="utf-8") as f:
                v0_data = json.load(f)
            sc_key = f"artape_v0_k{args.k}_lam{lam}_s{args.seed}"
            if sc_key in v0_data:
                in_train_ssim = v0_data[sc_key].get("final_adv_ssim_max", None)
        except Exception:
            pass

    gap_g = (res_2k["ssim"] - in_train_ssim) if in_train_ssim is not None else None

    print(f"\n[KẾT QUẢ FRESH {attacker_type.upper()} λ={lam}]")
    print(f"  - In-training Adv SSIM: {in_train_ssim:.4f}" if in_train_ssim else "  - In-training Adv SSIM: N/A")
    print(f"  - Fresh {attacker_type.upper()} (2k test): SSIM = {res_2k['ssim']:.4f} | PSNR = {res_2k['psnr']:.2f} dB | LPIPS = {res_2k['lpips']:.4f}")
    print(f"  - Fresh {attacker_type.upper()} (10k test): SSIM = {res_10k['ssim']:.4f} | PSNR = {res_10k['psnr']:.2f} dB")
    if gap_g is not None:
        print(f"  -> Discrepancy G = SSIM(Fresh) - SSIM(In-train) = {gap_g:+.4f}")
    print("-" * 85, flush=True)

    grid_path = os.path.join(args.output_dir, f"rec_artape_v0_fresh_{attacker_type}_lam{lam}_s{args.seed}.png")
    save_reconstruction_grid(client, attacker, test_eval_loader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=grid_path, num_images=8)

    return {
        "attacker_type": attacker_type,
        "lambda": lam,
        "seed": args.seed,
        "in_train_adv_ssim": in_train_ssim,
        "fresh_ssim_2k": round(res_2k["ssim"], 4),
        "fresh_psnr_2k": round(res_2k["psnr"], 2),
        "fresh_lpips_2k": round(res_2k["lpips"], 4) if res_2k["lpips"] else None,
        "fresh_mse_norm_2k": round(res_2k["mse_norm"], 4),
        "fresh_ssim_10k": round(res_10k["ssim"], 4),
        "fresh_psnr_10k": round(res_10k["psnr"], 2),
        "discrepancy_G": round(gap_g, 4) if gap_g is not None else None,
        "best_epoch": best_ep,
        "training_time_s": round(train_time, 1),
        "history": history
    }


def main():
    parser = argparse.ArgumentParser(description="Fresh Attacker Evaluation trên AR-TAPE v0 Client")
    parser.add_argument("--attackers", nargs="+", default=["mlp", "conv"], choices=["mlp", "conv"],
                        help="Danh sách attacker cần chạy: 'mlp', 'conv' hoặc cả hai")
    parser.add_argument("--k", type=int, default=1024, help="Chiều không gian con bottleneck k")
    parser.add_argument("--D", type=int, default=65536, help="Chiều phẳng smashed data")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Chiều ẩn MLP unprojector")
    parser.add_argument("--client-ckpt", type=str, default=None, help="Đường dẫn trực tiếp tới checkpoint client (nếu có)")
    parser.add_argument("--lambdas", nargs="+", type=float, default=None, help="Danh sách lambda")
    parser.add_argument("--lambda-val", type=float, default=None, help="Giá trị lambda đơn lẻ (alias tương thích)")
    parser.add_argument("--fixed-bottleneck", action="store_true", help="Cố định ma trận chiếu trực giao R tại client (Mode F)")
    parser.add_argument("--seed", type=int, default=42, help="Seed của checkpoint và data split")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện attacker (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam")
    parser.add_argument("--constant-lr", action="store_true", help="Giữ learning rate không đổi (không dùng CosineAnnealingLR)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_v0"))
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử 1 epoch rút gọn")
    args = parser.parse_args()

    if args.lambda_val is not None:
        if args.lambdas is None:
            args.lambdas = [args.lambda_val]
        else:
            args.lambdas.append(args.lambda_val)
    elif args.lambdas is None:
        args.lambdas = [0.1]

    if args.dry_run:
        args.epochs = 1
        args.eval_freq = 1
        args.lambdas = [0.1]
        args.output_dir = os.path.join(args.output_dir, "dry_run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

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
        client_ckpt_path = None
        if args.client_ckpt and os.path.isfile(args.client_ckpt):
            client_ckpt_path = args.client_ckpt
        else:
            candidates = [
                os.path.join(args.output_dir, "checkpoints", f"artape_v0_k{args.k}_linear_fixed_lam{lam}_s{args.seed}_client.pt"),
                os.path.join(args.output_dir, "checkpoints", f"artape_v0_k{args.k}_linear_fixed_fixR_lam{lam}_s{args.seed}_client.pt"),
                os.path.join(args.output_dir, "checkpoints", f"artape_v0_k{args.k}_linear_learned_lam{lam}_s{args.seed}_client.pt"),
                os.path.join(args.output_dir, "checkpoints", f"artape_v0_k{args.k}_lam{lam}_s{args.seed}_client.pt"),
                os.path.join(args.output_dir, f"artape_v0_k{args.k}_lam{lam}_s{args.seed}_client.pt"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    client_ckpt_path = c
                    break

        if not client_ckpt_path or not os.path.isfile(client_ckpt_path):
            print(f"[WARN] Bỏ qua λ={lam}: Không tìm thấy checkpoint client phù hợp tại {args.output_dir}")
            continue

        for atk in args.attackers:
            res = train_fresh_attacker(
                attacker_type=atk,
                client_ckpt_path=client_ckpt_path,
                lam=lam,
                args=args,
                train_loader=train_loader,
                test_eval_loader=test_eval_loader,
                test_full_loader=test_full_loader,
                device=device,
                lpips_fn=lpips_fn
            )
            all_results.append(res)

    json_path = os.path.join(args.output_dir, "results_artape_v0_fresh_attackers.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output_dir, "results_artape_v0_fresh_attackers.csv")
    if all_results:
        import csv
        keys = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

    print("\n" + "=" * 90)
    print("TỔNG HỢP KẾT QUẢ FRESH ATTACKERS TRÊN AR-TAPE v0:")
    print(f"{'Attacker':<10} | {'Lambda':<8} | {'In-train SSIM':<14} | {'Fresh SSIM (2k)':<16} | {'PSNR (2k)':<10} | {'LPIPS (2k)':<10} | {'Discrepancy G':<14}")
    print("-" * 90)
    for r in all_results:
        in_s = f"{r['in_train_adv_ssim']:.4f}" if r['in_train_adv_ssim'] is not None else "N/A"
        g_s = f"{r['discrepancy_G']:+.4f}" if r['discrepancy_G'] is not None else "N/A"
        lp_s = f"{r['fresh_lpips_2k']:.4f}" if r['fresh_lpips_2k'] is not None else "N/A"
        print(f"{r['attacker_type']:<10} | {r['lambda']:<8} | {in_s:<14} | {r['fresh_ssim_2k']:<16.4f} | {r['fresh_psnr_2k']:<10.2f} | {lp_s:<10} | {g_s:<14}")
    print("=" * 90)
    print(f"Đã lưu kết quả tại: {json_path} và {csv_path}")


if __name__ == "__main__":
    main()
