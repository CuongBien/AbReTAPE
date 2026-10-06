#!/usr/bin/env python3
# Bước 8: Baseline B8 - Min-Max Trực tuyến (Đặc tả 06/10/2026)
import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.data import get_cifar10, CIFAR10_MEAN, CIFAR10_STD
from src.attacks.decoder import Decoder
from src.metrics import psnr_ssim, distance_correlation
from src.utils.visualize import save_reconstruction_grid

class LearnedAdaptiveDecoder(nn.Module):
    """
    Adversary B8-mlp: Học mạng MLP unprojector 65536 -> hidden_dim -> 65536,
    sau đó reshape và dùng Conv Decoder. Dùng cho kịch bản adversary rất mạnh.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 8: B8 Min-Max Trực Tuyến")
    parser.add_argument("--lam", type=float, default=0.5, help="Hệ số lambda (mặc định: 0.5)")
    parser.add_argument("--cap", type=float, default=-1.0, help="Ngưỡng chặn MSE (cap). Dùng -1 để tự động tính.")
    parser.add_argument("--adv-type", type=str, choices=["conv", "mlp"], default="conv", help="Loại Adversary (conv hoặc mlp)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate Client/Server (mặc định: 0.1)")
    parser.add_argument("--adv-lr", type=float, default=1e-3, help="Learning rate Adversary (mặc định: 1e-3)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step8"))
    return parser.parse_args()


def calculate_cap(loader, device):
    """Tính toán ngưỡng MSE tự động bằng phương sai trung bình của ảnh (sai số khi đoán bằng ảnh trung bình)"""
    sum_img = None
    total_samples = 0
    # Tính ảnh trung bình
    for x, _ in loader:
        if sum_img is None:
            sum_img = torch.zeros_like(x[0])
        sum_img += x.sum(dim=0)
        total_samples += x.size(0)
    
    mean_img = (sum_img / total_samples).to(device)
    
    # Tính MSE
    sum_mse = 0.0
    for x, _ in loader:
        x = x.to(device)
        mse = F.mse_loss(mean_img.expand_as(x), x, reduction='sum')
        sum_mse += mse.item()
        
    avg_mse = sum_mse / (total_samples * x.size(1) * x.size(2) * x.size(3))
    return avg_mse


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"BƯỚC 8: B8 MIN-MAX TRỰC TUYẾN (KHÔNG KHÁNG HẤP THỤ)")
    print(f"Thiết bị: {device}")
    print(f"Adversary: {args.adv_type} | Lambda: {args.lam}")
    print("=" * 80)
    
    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # 1. Tính toán Cap nếu cần
    cap = args.cap
    if cap < 0:
        print("[INIT] Đang tính toán giá trị cap từ tập huấn luyện...")
        cap = calculate_cap(trainloader, device)
        print(f"[INIT] Đã tính được cap tự động: {cap:.4f}")
    
    # 2. Khởi tạo mô hình
    client = ClientModel().to(device)
    server = ServerModel().to(device)
    
    if args.adv_type == "conv":
        adv = Decoder(in_channels=64, out_channels=3).to(device)
    else:
        adv = LearnedAdaptiveDecoder(in_features=65536, D=65536, hidden_dim=512).to(device)
        
    # 3. Optimizers
    opt_cs = torch.optim.SGD(list(client.parameters()) + list(server.parameters()), 
                             lr=args.lr, momentum=0.9, weight_decay=5e-4)
    opt_adv = torch.optim.Adam(adv.parameters(), lr=args.adv_lr)
    
    sched_cs = MultiStepLR(opt_cs, milestones=[50, 75], gamma=0.1)
    
    # Lấy 2000 test images tĩnh để đánh giá adversary nội tại
    test_images, test_labels = [], []
    for x, y in testloader:
        test_images.append(x)
        test_labels.append(y)
        if sum([t.size(0) for t in test_images]) >= 2000:
            break
    fixed_test_x = torch.cat(test_images)[:2000].to(device)
    fixed_test_y = torch.cat(test_labels)[:2000].to(device)
    
    # 4. Huấn luyện
    print("\n[BẮT ĐẦU HUẤN LUYỆN MIN-MAX]")
    history = []
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        client.train()
        server.train()
        adv.train()
        
        total_task_loss, total_adv_loss, total_rec = 0.0, 0.0, 0.0
        n_samples = 0
        
        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            b = x.size(0)
            
            # --- BƯỚC MAX: Adversary học giải mã z hiện tại ---
            for p in adv.parameters():
                p.requires_grad = True
                
            with torch.no_grad():
                z_det = client(x)
                
            x_hat_adv = adv(z_det)
            loss_adv = F.mse_loss(x_hat_adv, x)
            
            opt_adv.zero_grad()
            loss_adv.backward()
            opt_adv.step()
            
            # --- BƯỚC MIN: Client & Server cập nhật ---
            for p in adv.parameters():
                p.requires_grad = False
                
            z = client(x)
            logits = server(z)
            loss_task = F.cross_entropy(logits, y)
            
            x_hat_cs = adv(z)
            rec = F.mse_loss(x_hat_cs, x)
            
            # Hàm mất mát tổng (có kẹp rec)
            loss = loss_task - args.lam * torch.clamp(rec, max=cap)
            
            opt_cs.zero_grad()
            loss.backward()
            opt_cs.step()
            
            total_task_loss += loss_task.item() * b
            total_adv_loss += loss_adv.item() * b
            total_rec += rec.item() * b
            n_samples += b
            
        sched_cs.step()
        ep_time = time.time() - t0
        
        # --- Đánh giá ---
        client.eval()
        server.eval()
        adv.eval()
        
        with torch.no_grad():
            z_test = client(fixed_test_x)
            logits_test = server(z_test)
            acc = (logits_test.argmax(1) == fixed_test_y).float().mean().item()
            
            x_hat_test = adv(z_test)
            p, s = psnr_ssim(fixed_test_x, x_hat_test, CIFAR10_MEAN, CIFAR10_STD)
            
        print(f"Epoch {epoch:3d}/{args.epochs} ({ep_time:.1f}s) | Task CE: {total_task_loss/n_samples:.4f} | "
              f"Adv MSE: {total_adv_loss/n_samples:.4f} | Acc: {acc*100:.2f}% | Adv PSNR: {p:.2f}dB | Adv SSIM: {s:.4f}")
              
        hist_entry = {
            "epoch": epoch,
            "task_ce": total_task_loss/n_samples,
            "adv_mse": total_adv_loss/n_samples,
            "acc": acc,
            "adv_psnr": p,
            "adv_ssim": s
        }
        
        # Đo dCor mỗi 10 epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            dcor_sum = 0.0
            n_dcor = 0
            with torch.no_grad():
                for bx, _ in testloader:
                    bx = bx.to(device)
                    bz = client(bx)
                    dcor_sum += distance_correlation(bx, bz).item()
                    n_dcor += 1
                    if n_dcor >= 15:
                        break
            hist_entry["dcor"] = dcor_sum / max(n_dcor, 1)
            print(f"  --> dCor(X, Z): {hist_entry['dcor']:.4f}")
            
        history.append(hist_entry)
        
    # Lưu checkpoint
    ckpt_path = os.path.join(args.output_dir, f"b8_{args.adv_type}_lam_{args.lam}.pt")
    torch.save({
        "client": client.state_dict(),
        "server": server.state_dict(),
        "adv": adv.state_dict(),
        "args": vars(args),
        "history": history
    }, ckpt_path)
    
    print(f"\n[DONE] Hoàn tất huấn luyện B8! Đã lưu checkpoint tại {ckpt_path}")
    
    json_path = os.path.join(args.output_dir, f"b8_{args.adv_type}_lam_{args.lam}_history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
