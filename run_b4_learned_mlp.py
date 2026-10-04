#!/usr/bin/env python3
# ==============================================================================
# ĐO ĐẠC FRESH ATTACKER MẠNH NHẤT: LEARNED ADAPTIVE MLP TRÊN CLIENT B4 ĐÃ ĐÓNG BĂNG
# ==============================================================================
# Huấn luyện Learned MLP Unprojector (65536 -> 512 -> 65536) + Conv Decoder
# trên Client B4 (Block Scramble BS=4) đã đóng băng hoàn toàn.
# Nhằm lấy số liệu cho Attacker Fresh mạnh nhất của B4 đưa vào Bảng 1.
# ==============================================================================
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
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel
from src.data import get_cifar10, CIFAR10_MEAN, CIFAR10_STD
from src.defenses import BlockScrambleDefense
from src.attacks import Decoder
from src.metrics import psnr_ssim, get_lpips_fn, calculate_lpips, distance_correlation
from src.utils.visualize import save_reconstruction_grid


class LearnedAdaptiveDecoderB4(nn.Module):
    """
    Attacker 2 (Kerckhoffs Strong Adaptive Attacker):
    Học mạng MLP unprojector 65536 -> hidden_dim -> 65536 để đảo ngược phép xáo trộn khối,
    sau đó reshape về (B, 64, 32, 32) và đưa vào Conv Decoder.
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
            z = z.view(z.size(0), -1)
        z_unproj = self.unprojector(z).view(-1, 64, 32, 32)
        return self.conv_decoder(z_unproj)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Fresh Learned MLP Attacker trên B4 đã đóng băng")
    parser.add_argument("--block-size", type=int, default=4, help="Block size của B4 (mặc định: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Seed hoán vị của B4 (mặc định: 42)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện decoder (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam (mặc định: 1e-3)")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Chiều ẩn MLP (mặc định: 512)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá (mặc định: 5)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step3_B4"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 85)
    print("HUẤN LUYỆN FRESH ATTACKER MẠNH NHẤT: LEARNED ADAPTIVE MLP TRÊN CLIENT B4 ĐÃ ĐÓNG BĂNG")
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Baseline: B4 Block Scramble (BS={args.block_size}, Seed={args.seed})")
    print(f"Kiến trúc Attacker: MLP Unprojector ({65536} -> {args.hidden_dim} -> {65536}) + ConvDecoder")
    print(f"Số epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print("=" * 85, flush=True)

    # 1. Nạp CIFAR-10 full-train
    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    lpips_fn = get_lpips_fn(device=device)

    # 2. Nạp Client B4 đã đóng băng
    client = ClientModel().to(device)
    b4_ckpt_path = os.path.join(args.output_dir, f"b4_best_block_size_{args.block_size}.pt")
    if os.path.isfile(b4_ckpt_path):
        print(f"[LOAD] Nạp checkpoint Client B4 từ: {b4_ckpt_path}")
        ckpt = torch.load(b4_ckpt_path, map_location=device)
        client.load_state_dict(ckpt["client"])
    else:
        print(f"[WARN] Không tìm thấy {b4_ckpt_path}, dùng Client khởi tạo ngẫu nhiên")

    client.eval()
    for p in client.parameters():
        p.requires_grad = False

    defense = BlockScrambleDefense(block_size=args.block_size, seed=args.seed).to(device)
    defense.eval()

    # Đo dCor trên test set
    print("[METRIC] Đang tính toán dCor(X, Z_scrambled)...", flush=True)
    dcor_sum = 0.0
    n_dcor = 0
    with torch.no_grad():
        for x_t, _ in testloader:
            x_t = x_t.to(device)
            z_t = defense(client(x_t))
            dcor_sum += distance_correlation(x_t, z_t).item()
            n_dcor += 1
            if n_dcor >= 15:
                break
    avg_dcor = dcor_sum / max(n_dcor, 1)
    print(f"  -> Distance Correlation dCor: {avg_dcor:.4f}")

    # 3. Khởi tạo Learned Adaptive Decoder
    decoder = LearnedAdaptiveDecoderB4(in_features=65536, D=65536, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_psnr = 0.0
    best_ssim = 0.0
    best_ep = 0

    print(f"\n[BẮT ĐẦU] Huấn luyện Learned Adaptive Decoder ({args.epochs} Epochs)...", flush=True)
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        decoder.train()
        total_loss = 0.0
        n_samples = 0

        for x, _ in trainloader:
            x = x.to(device, non_blocking=True)
            b = x.size(0)

            with torch.no_grad():
                z = defense(client(x))
            z_input = z.detach()

            optimizer.zero_grad()
            x_hat = decoder(z_input)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b
            n_samples += b

        scheduler.step()
        train_mse = total_loss / n_samples

        if ep % args.eval_freq == 0 or ep == args.epochs:
            decoder.eval()
            psnr_sum, ssim_sum, total_test = 0.0, 0.0, 0
            with torch.no_grad():
                for x_val, _ in testloader:
                    x_val = x_val.to(device, non_blocking=True)
                    bv = x_val.size(0)
                    z_val = defense(client(x_val))
                    x_rec = decoder(z_val)
                    p, s = psnr_ssim(x_val, x_rec, CIFAR10_MEAN, CIFAR10_STD)
                    psnr_sum += p * bv
                    ssim_sum += s * bv
                    total_test += bv

            cur_psnr = psnr_sum / total_test
            cur_ssim = ssim_sum / total_test

            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                best_ssim = cur_ssim
                best_ep = ep
                # Lưu checkpoint tốt nhất
                ckpt_save_path = os.path.join(args.output_dir, f"b4_decoder_learned_mlp_bs{args.block_size}.pt")
                torch.save({"decoder": decoder.state_dict(), "best_psnr": best_psnr, "best_ssim": best_ssim, "epoch": ep}, ckpt_save_path)

            print(f"  [Epoch {ep:2d}/{args.epochs}] Train MSE: {train_mse:.4f} | Test PSNR: {cur_psnr:.2f} dB | SSIM: {cur_ssim:.4f} (Best: {best_psnr:.2f} dB Ep {best_ep})", flush=True)

    t_total = time.time() - t_start
    print(f"\n[HOÀN TẤT] Huấn luyện xong sau {t_total:.1f}s!")

    # Đánh giá toàn diện cuối cùng với LPIPS
    decoder.eval()
    psnr_sum, ssim_sum, lpips_sum, mse_sum, total_test = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x_val, _ in testloader:
            x_val = x_val.to(device, non_blocking=True)
            bv = x_val.size(0)
            z_val = defense(client(x_val))
            x_rec = decoder(z_val)
            loss_val = criterion(x_rec, x_val)
            p, s = psnr_ssim(x_val, x_rec, CIFAR10_MEAN, CIFAR10_STD)
            lp = calculate_lpips(x_val, x_rec, CIFAR10_MEAN, CIFAR10_STD, lpips_fn)

            mse_sum += loss_val.item() * bv
            psnr_sum += p * bv
            ssim_sum += s * bv
            if lp is not None:
                lpips_sum += lp * bv
            total_test += bv

    final_mse = mse_sum / total_test
    final_psnr = psnr_sum / total_test
    final_ssim = ssim_sum / total_test
    final_lpips = lpips_sum / total_test if lpips_fn else None

    # Xuất ảnh tái tạo
    grid_path = os.path.join(args.output_dir, f"b4_reconstruction_learned_mlp_bs{args.block_size}.png")
    save_reconstruction_grid(client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=grid_path, num_images=8, defense=defense)

    result_data = {
        "defense": "B4",
        "method": "Block Scrambling (Learned Adaptive Attacker)",
        "block_size": args.block_size,
        "seed": args.seed,
        "attacker": "learned_adaptive_mlp",
        "test_acc": 0.9261, # từ client B4 đã đóng băng
        "mse": round(final_mse, 4),
        "psnr": round(final_psnr, 2),
        "ssim": round(final_ssim, 4),
        "lpips": round(final_lpips, 4) if final_lpips else None,
        "dcor": round(avg_dcor, 4),
        "best_psnr": round(best_psnr, 2),
        "best_ssim": round(best_ssim, 4),
        "best_epoch": best_ep,
        "training_time_s": round(t_total, 1),
    }

    res_json_path = os.path.join(args.output_dir, f"results_b4_learned_mlp_bs{args.block_size}.json")
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print("\n" + "=" * 85)
    print("KẾT QUẢ ĐỐI SÁNH FRESH ATTACKER TRÊN B4 (BS=4):")
    print(f"  * Conv Decoder cũ (Attacker 1):      PSNR = 15.62 dB | SSIM = 0.3015 | LPIPS = 0.1269")
    print(f"  * Learned Adaptive MLP (Attacker 2):  PSNR = {final_psnr:.2f} dB | SSIM = {final_ssim:.4f} | LPIPS = {final_lpips:.4f if final_lpips else 'N/A'}")
    print(f"  * Distance Correlation dCor:         {avg_dcor:.4f}")
    print(f"  * Checkpoint đã lưu:                 {os.path.join(args.output_dir, f'b4_decoder_learned_mlp_bs{args.block_size}.pt')}")
    print(f"  * Lưới ảnh tái tạo:                  {grid_path}")
    print("=" * 85 + "\n", flush=True)


if __name__ == "__main__":
    main()
