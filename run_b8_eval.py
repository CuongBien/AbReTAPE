#!/usr/bin/env python3
# Bước 8: Đánh giá hậu kiểm (Post-training Attacks) trên Baseline B8
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
from src.attacks.decoder import Decoder
from src.metrics import psnr_ssim, calculate_lpips, get_lpips_fn
from run_b8_minmax import LearnedAdaptiveDecoder

def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá hậu kiểm B8: N1 Decision")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn checkpoint B8")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện fresh attackers (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step8_Eval"))
    return parser.parse_args()

def train_fresh_attacker(attacker_name, attacker_model, client, trainloader, testloader, args, device, lpips_fn):
    print(f"\n--- HUẤN LUYỆN FRESH ATTACKER: {attacker_name} ---")
    opt = torch.optim.Adam(attacker_model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    best_psnr, best_ssim = 0.0, 0.0
    
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        attacker_model.train()
        for x, _ in trainloader:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                z = client(x)
                
            opt.zero_grad()
            x_rec = attacker_model(z)
            loss = criterion(x_rec, x)
            loss.backward()
            opt.step()
        sched.step()
        
        # Eval
        attacker_model.eval()
        psnr_sum, ssim_sum, total = 0.0, 0.0, 0
        with torch.no_grad():
            for x_val, _ in testloader:
                x_val = x_val.to(device)
                b = x_val.size(0)
                z_val = client(x_val)
                x_rec = attacker_model(z_val)
                p, s = psnr_ssim(x_val, x_rec, CIFAR10_MEAN, CIFAR10_STD)
                psnr_sum += p * b
                ssim_sum += s * b
                total += b
                
        cur_psnr = psnr_sum / total
        cur_ssim = ssim_sum / total
        if cur_psnr > best_psnr:
            best_psnr = cur_psnr
            best_ssim = cur_ssim
            
        print(f"[{attacker_name}] Epoch {ep}/{args.epochs} | PSNR: {cur_psnr:.2f}dB | SSIM: {cur_ssim:.4f}")
        
    print(f"Xong {attacker_name} sau {time.time() - t0:.1f}s. Best PSNR: {best_psnr:.2f}dB, Best SSIM: {best_ssim:.4f}")
    return best_psnr, best_ssim

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("BƯỚC 8: ĐÁNH GIÁ HẬU KIỂM & KIỂM CHỨNG GIẢ THUYẾT N1")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)
    
    # Nạp Client B8
    client = ClientModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    client.load_state_dict(ckpt["client"])
    client.eval()
    for p in client.parameters():
        p.requires_grad = False
        
    # Lấy thông tin adv trong quá trình huấn luyện
    hist = ckpt.get("history", [])
    adv_train_ssim = hist[-1]["adv_ssim"] if hist else 0.0
    adv_train_psnr = hist[-1]["adv_psnr"] if hist else 0.0
    print(f"[INFO] Adversary nội tại trong huấn luyện: PSNR = {adv_train_psnr:.2f}dB, SSIM = {adv_train_ssim:.4f}")
    
    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    lpips_fn = get_lpips_fn(device)
    
    # 1. Fresh Conv Decoder
    conv_dec = Decoder(in_channels=64, out_channels=3).to(device)
    psnr_conv, ssim_conv = train_fresh_attacker("Fresh Conv Decoder", conv_dec, client, trainloader, testloader, args, device, lpips_fn)
    
    # 2. Fresh Learned MLP
    mlp_dec = LearnedAdaptiveDecoder(in_features=65536, D=65536, hidden_dim=512).to(device)
    psnr_mlp, ssim_mlp = train_fresh_attacker("Fresh Learned MLP", mlp_dec, client, trainloader, testloader, args, device, lpips_fn)
    
    # Tính toán G
    G = ssim_mlp - adv_train_ssim
    
    print("\n" + "=" * 80)
    print("BẢNG QUYẾT ĐỊNH N1")
    print("=" * 80)
    print(f"1. SSIM(Adv trong huấn luyện) : {adv_train_ssim:.4f}")
    print(f"2. SSIM(Fresh Conv Decoder)   : {ssim_conv:.4f}")
    print(f"3. SSIM(Fresh Learned MLP)    : {ssim_mlp:.4f}")
    print(f"--> Độ đo chính (G)           : {G:+.4f}")
    print("-" * 80)
    
    if adv_train_ssim < 0.3 and (ssim_mlp > 0.3 or ssim_conv > 0.3) and G > 0.05:
        print("[KẾT LUẬN]: Ủng hộ N1. Min-max trực tuyến bị đánh lừa vì adversary nội tại yếu hơn attacker hậu kiểm.")
    elif ssim_mlp < 0.3 and ssim_conv < 0.3 and abs(G) <= 0.05:
        print("[KẾT LUẬN]: B8 đã đủ mạnh. N1 sai trên cấu hình này (cả attacker nội tại và hậu kiểm đều thất bại).")
        print("Cần xem lại cơ chế AR-TAPE thực sự vượt B8 ở điểm nào (có thể là kháng FSHA hoặc acc).")
    else:
        print("[KẾT LUẬN]: Kết quả nằm ở vùng trung gian, phụ thuộc vào sức mạnh cụ thể của cấu hình.")
    print("=" * 80)

if __name__ == "__main__":
    main()
