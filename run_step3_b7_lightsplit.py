#!/usr/bin/env python3
# ==============================================================================
# BƯỚC 3: BASELINE B7 — FIXED ORTHOGONAL PROJECTION (LIGHTSPLIT-STYLE, arXiv:2605.13265)
# ==============================================================================
# Cơ chế phòng thủ chiếu trực giao cố định tại Cut-Layer 1:
# - Client: z in R^{B x 65536} -> z_t = R^T z in R^{B x k} (k in {512, 1024, 2048})
# - Server Mode F: z_hat = R z_t (0 tham số bổ sung) -> reshape (64, 32, 32) -> Server B0
# - Server Mode L: MLP k -> m -> 65536 có BN -> reshape (64, 32, 32) -> Server B0
# - Threat Model: Attacker biết ma trận R, lift z_hat = R z_t và huấn luyện Decoder
# ==============================================================================
import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

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
from src.defenses import FixedOrthoProjection, SplitProjection, wcc_loss
from src.attacks import Decoder
from src.metrics import psnr_ssim, get_lpips_fn, calculate_lpips, distance_correlation
from src.utils import save_reconstruction_grid

D_DIM = 64 * 32 * 32  # 65,536


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline B7: LightSplit-style Fixed Orthogonal Projection")
    parser.add_argument("--k", type=int, default=1024, choices=[256, 512, 1024, 2048, 4096],
                        help="Số chiều không gian con chiếu k (mặc định: 1024, CR=64x)")
    parser.add_argument("--mode", type=str, default="F", choices=["F", "L"],
                        help="Chế độ server: F (không tham số) hoặc L (MLP có BN, mặc định: F)")
    parser.add_argument("--m", type=int, default=512, help="Chiều ẩn MLP trong chế độ L (mặc định: 512)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed sinh ma trận trực chuẩn R (mặc định: 42)")
    parser.add_argument("--train-seed", type=int, default=0, help="Random seed cho huấn luyện SL (mặc định: 0)")
    parser.add_argument("--wcc-lam", type=float, default=0.0,
                        help="Hệ số WCC loss (mặc định: 0.0 do testbed chuẩn nhãn ở server)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện Split Learning (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate cho SGD (mặc định: 0.1)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "multistep"],
                        help="Bộ điều chỉnh LR: cosine hoặc multistep ([50, 75])")
    parser.add_argument("--decoder-epochs", type=int, default=30, help="Số epochs huấn luyện Decoder tấn công (mặc định: 30)")
    parser.add_argument("--decoder-lr", type=float, default=1e-3, help="Learning rate cho Decoder Adam (mặc định: 1e-3)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá test accuracy (mặc định: 5)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2,
                        help="Số worker nạp dữ liệu")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục dữ liệu")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step3_B7"),
                        help="Thư mục lưu outputs")
    parser.add_argument("--resume", action="store_true", default=False, help="Khôi phục tiến trình nếu có")
    return parser.parse_args()


def evaluate_sl(client, proj, server, testloader, device):
    client.eval()
    server.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            z = client(x)
            zt = proj(z)
            logits = server(zt)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return correct / max(total, 1)


def train_sl_epoch(client, proj, server, trainloader, opt_c, opt_s, device, wcc_lam=0.0):
    client.train()
    server.train()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in trainloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        b = x.size(0)

        # 1. Forward client + projection
        z = client(x)
        zt = proj(z)

        # 2. Gradient cut-point giữa Client và Server
        zt_detached = zt.detach().requires_grad_(True)
        logits = server(zt_detached)
        loss = ce(logits, y)

        # 3. Backward Server
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()

        # 4. Backward Client qua phép chiếu trực giao
        opt_c.zero_grad()
        zt.backward(zt_detached.grad)

        # Biến thể nếu có dùng WCC loss
        if wcc_lam > 0.0:
            loss_wcc = wcc_loss(zt, y)
            (wcc_lam * loss_wcc).backward()

        opt_c.step()

        total_loss += loss.item() * b
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += b

    return total_loss / total, correct / total


def train_decoder_attack(decoder, client, proj, trainloader, testloader, epochs, lr, device, eval_freq=5):
    print("\n" + "=" * 70)
    print(f"HUẤN LUYỆN DECODER TẤN CÔNG (Attacker biết ma trận R, lift z_hat = R z_t)")
    print(f"Tổng số epochs: {epochs} | LR: {lr} (Adam) | Loss: MSE")
    print("=" * 70)

    decoder.train()
    client.eval()
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss, total_samples = 0.0, 0

        for x, _ in trainloader:
            x = x.to(device, non_blocking=True)
            b = x.size(0)

            with torch.no_grad():
                z = client(x)
                zt = proj(z)
                # Attacker lift: z_hat = R z_t, reshape (B, 64, 32, 32)
                z_hat = proj.lift(zt).view(b, 64, 32, 32)

            x_rec = decoder(z_hat)
            loss = mse(x_rec, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * b
            total_samples += b

        dt = time.time() - t0
        avg_loss = total_loss / total_samples

        if epoch % eval_freq == 0 or epoch == epochs:
            print(f"[Decoder Attack] Epoch {epoch:2d}/{epochs:2d} ({dt:.1f}s) | Train MSE: {avg_loss:.5f}")

    return decoder


def evaluate_decoder_attack(decoder, client, proj, testloader, device, lpips_fn=None):
    decoder.eval()
    client.eval()
    mse_fn = nn.MSELoss()

    total_mse, total_psnr, total_ssim, total_lpips, total_samples = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device, non_blocking=True)
            b = x.size(0)
            z = client(x)
            zt = proj(z)
            z_hat = proj.lift(zt).view(b, 64, 32, 32)

            x_rec = decoder(z_hat)
            loss_mse = mse_fn(x_rec, x).item()
            p, s = psnr_ssim(x, x_rec, CIFAR10_MEAN, CIFAR10_STD)

            total_mse += loss_mse * b
            total_psnr += p * b
            total_ssim += s * b
            if lpips_fn is not None:
                lp_val = calculate_lpips(x, x_rec, CIFAR10_MEAN, CIFAR10_STD, lpips_fn)
                total_lpips += lp_val * b
            total_samples += b

    avg_mse = total_mse / total_samples
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    avg_lpips = (total_lpips / total_samples) if lpips_fn is not None else None
    return avg_mse, avg_psnr, avg_ssim, avg_lpips


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = f"k{args.k}_{args.mode}_s{args.seed}"
    sl_ckpt_path = os.path.join(args.output_dir, f"b7_sl_{tag}.pt")
    dec_ckpt_path = os.path.join(args.output_dir, f"b7_decoder_{tag}.pt")
    grid_img_path = os.path.join(args.output_dir, f"b7_reconstruction_{tag}.png")
    json_path = os.path.join(args.output_dir, f"results_b7_{tag}.json")

    print("\n" + "=" * 80)
    print("BƯỚC 3 — BASELINE B7: LIGHTSPLIT FIXED ORTHOGONAL PROJECTION (arXiv:2605.13265)")
    print(f"Device: {device} | k={args.k} (CR={D_DIM//args.k}x) | Mode: {args.mode} | Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # 1. Khởi tạo ma trận trực giao R
    print("\n==> [BƯỚC 0] Khởi tạo ma trận trực chuẩn R (QR decomposition)...")
    proj = FixedOrthoProjection(D=D_DIM, k=args.k, seed=args.seed, device=device).to(device)
    ortho_err = proj.get_orthogonality_error()
    mem_gib = proj.get_memory_bytes() / (1024 ** 3)
    print(f"  -> Kích thước R: ({proj.D}, {proj.k})")
    print(f"  -> Dung lượng VRAM của R: {mem_gib:.2f} GiB")
    print(f"  -> Độ lệch trực chuẩn max |R^T R - I_k|: {ortho_err:.2e} (chuẩn: ~0)")

    # 2. Chuẩn bị dữ liệu
    print(f"\n==> [DỮ LIỆU] Nạp CIFAR-10 (Batch: {args.batch_size}, Workers: {args.num_workers})...")
    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # 3. Xây dựng Client & Server (SplitProjection)
    client = ClientModel().to(device)
    base_server = ServerModel().to(device)
    server = SplitProjection(proj, base_server, mode=args.mode, m=args.m).to(device)

    # =========================================================================
    # GIAI ĐOẠN 1: HUẤN LUYỆN SPLIT LEARNING VỚI PHÉP CHIẾU
    # =========================================================================
    if args.resume and os.path.isfile(sl_ckpt_path):
        print(f"\n[RESUME] Tìm thấy checkpoint SL tại {sl_ckpt_path}. Đang nạp...")
        ckpt = torch.load(sl_ckpt_path, map_location=device)
        client.load_state_dict(ckpt["client"])
        server.load_state_dict(ckpt["server"])
        test_acc = evaluate_sl(client, proj, server, testloader, device)
        print(f"[RESUME] Nạp thành công. Test Accuracy: {test_acc*100:.2f}%")
    else:
        torch.manual_seed(args.train_seed)
        opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        if args.scheduler == "cosine":
            sched_c = CosineAnnealingLR(opt_c, T_max=args.epochs)
            sched_s = CosineAnnealingLR(opt_s, T_max=args.epochs)
        else:
            sched_c = MultiStepLR(opt_c, milestones=[50, 75], gamma=0.1)
            sched_s = MultiStepLR(opt_s, milestones=[50, 75], gamma=0.1)

        print("\n" + "=" * 70)
        print(f"GIAI ĐOẠN 1: HUẤN LUYỆN SPLIT LEARNING VỚI PROJECTION ({args.epochs} Epochs)")
        print("=" * 70)

        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss, acc = train_sl_epoch(client, proj, server, trainloader, opt_c, opt_s, device, wcc_lam=args.wcc_lam)
            sched_c.step()
            sched_s.step()
            dt = time.time() - t0

            if epoch % args.eval_freq == 0 or epoch == args.epochs:
                t_acc = evaluate_sl(client, proj, server, testloader, device)
                if t_acc > best_acc:
                    best_acc = t_acc
                print(f"[B7 SL] Epoch {epoch:3d}/{args.epochs:3d} ({dt:.1f}s) | Train Loss: {loss:.4f} | Train Acc: {acc*100:.2f}% | Test Acc: {t_acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

        test_acc = evaluate_sl(client, proj, server, testloader, device)
        torch.save({
            "client": client.state_dict(),
            "server": server.state_dict(),
            "k": args.k,
            "mode": args.mode,
            "seed": args.seed,
            "test_acc": test_acc
        }, sl_ckpt_path)
        print(f"[B7] Đã lưu checkpoint SL tại: {sl_ckpt_path} với Test Acc: {test_acc*100:.2f}%")

    # =========================================================================
    # GIAI ĐOẠN 2: HUẤN LUYỆN DECODER TẤN CÔNG (ATTACKER BIẾT R)
    # =========================================================================
    decoder = Decoder().to(device)
    if args.resume and os.path.isfile(dec_ckpt_path):
        print(f"\n[RESUME] Tìm thấy checkpoint Decoder tại {dec_ckpt_path}. Đang nạp...")
        decoder.load_state_dict(torch.load(dec_ckpt_path, map_location=device))
    else:
        decoder = train_decoder_attack(
            decoder, client, proj, trainloader, testloader,
            epochs=args.decoder_epochs, lr=args.decoder_lr, device=device, eval_freq=args.eval_freq
        )
        torch.save(decoder.state_dict(), dec_ckpt_path)
        print(f"[B7] Đã lưu checkpoint Decoder tại: {dec_ckpt_path}")

    # =========================================================================
    # GIAI ĐOẠN 3: ĐÁNH GIÁ METRICS & XUẤT ẢNH TÁI TẠO
    # =========================================================================
    print("\n==> [ĐÁNH GIÁ] Đang tính toán PSNR, SSIM, LPIPS và dCor...")
    lpips_fn = get_lpips_fn(device=device)
    dec_mse, dec_psnr, dec_ssim, dec_lpips = evaluate_decoder_attack(
        decoder, client, proj, testloader, device, lpips_fn=lpips_fn
    )

    # Đo khoảng cách tương quan dCor(X, Z_hat) và dCor(X, Z_t)
    client.eval()
    dcor_zhat_sum, dcor_zt_sum, n_eval = 0.0, 0.0, 0
    with torch.no_grad():
        for x_t, _ in testloader:
            x_t = x_t.to(device)
            z = client(x_t)
            zt = proj(z)
            z_hat = proj.lift(zt).view(x_t.size(0), 64, 32, 32)

            dcor_zhat_sum += distance_correlation(x_t, z_hat).item()
            dcor_zt_sum += distance_correlation(x_t.view(x_t.size(0), -1), zt).item()
            n_eval += 1
            if n_eval >= 15:
                break
    dcor_zhat = dcor_zhat_sum / max(n_eval, 1)
    dcor_zt = dcor_zt_sum / max(n_eval, 1)

    # Xuất lưới ảnh tái tạo
    class LightSplitWrapper(nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p
        def forward(self, z):
            b = z.size(0)
            zt = self.p(z)
            return self.p.lift(zt).view(b, 64, 32, 32)

    save_reconstruction_grid(
        client, decoder, testloader, device,
        CIFAR10_MEAN, CIFAR10_STD,
        defense=LightSplitWrapper(proj),
        save_path=grid_img_path, num_images=8
    )

    # =========================================================================
    # TỔNG KẾT KẾT QUẢ VÀ LƯU JSON
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ TỔNG KẾT BASELINE B7: LIGHTSPLIT (k={args.k}, Mode={args.mode})")
    print("=" * 80)
    print(f"Test Accuracy:         {test_acc*100:.2f}% (B0 Vanilla: 94.71% | Delta: {(test_acc - 0.9471)*100:+.2f}%)")
    print(f"Reconstruction PSNR:   {dec_psnr:.2f} dB (B0: 40.93 dB | Delta: {dec_psnr - 40.93:+.2f} dB)")
    print(f"Reconstruction SSIM:   {dec_ssim:.4f} (B0: 0.9955)")
    print(f"Reconstruction LPIPS:  {dec_lpips:.4f} (B0: 0.0004)")
    print(f"Information Leakage:   dCor(X, Z_hat)={dcor_zhat:.4f} | dCor(X, Z_t)={dcor_zt:.4f}")
    print(f"Lưới ảnh đối chứng:    {grid_img_path}")
    print("=" * 80)

    res_data = {
        "defense": "B7_LightSplit",
        "k": args.k,
        "compression_ratio": D_DIM // args.k,
        "mode": args.mode,
        "seed": args.seed,
        "orthogonality_error": ortho_err,
        "vram_gib": mem_gib,
        "test_acc": test_acc,
        "mse": dec_mse,
        "psnr": dec_psnr,
        "ssim": dec_ssim,
        "lpips": dec_lpips,
        "dcor_zhat": dcor_zhat,
        "dcor_zt": dcor_zt,
        "dcor": dcor_zhat
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res_data, f, indent=2)
    print(f"[LƯU] Kết quả JSON chi tiết: {json_path}")

    # Cập nhật vào full_comprehensive_evaluation.json
    full_eval_path = os.path.join(PROJECT_ROOT, "output", "full_comprehensive_evaluation.json")
    if os.path.isfile(full_eval_path):
        try:
            with open(full_eval_path, "r", encoding="utf-8") as f:
                full_eval = json.load(f)
            key_name = f"B7_LightSplit_k_{args.k}_{args.mode}_s{args.seed}"
            full_eval[key_name] = {
                "method": f"B7: LightSplit (Fixed Orthogonal Proj, Mode {args.mode})",
                "config": f"k={args.k} (CR={D_DIM//args.k}x), seed={args.seed}",
                "test_acc": test_acc,
                "mse": dec_mse,
                "psnr": dec_psnr,
                "ssim": dec_ssim,
                "lpips": dec_lpips,
                "dcor": dcor_zhat
            }
            with open(full_eval_path, "w", encoding="utf-8") as f:
                json.dump(full_eval, f, indent=2, ensure_ascii=False)
            print(f"[CẬP NHẬT] Đã ghi nhận {key_name} vào full_comprehensive_evaluation.json")
        except Exception as e:
            print(f"[WARN] Không thể cập nhật full_comprehensive_evaluation.json: {e}")


if __name__ == "__main__":
    main()
