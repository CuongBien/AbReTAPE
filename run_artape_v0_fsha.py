#!/usr/bin/env python3
# ==============================================================================
# BƯỚC 4: TẤN CÔNG CHỦ ĐỘNG FSHA TRÊN CLIENT AR-TAPE v0 (RES-SFL-LITE)
# (Feature Space Hijacking Attack - Pasquini et al., CCS 2021)
# ==============================================================================
# - Mục tiêu: Đánh giá khả năng kháng thao túng gradient của Client AR-TAPE v0
#   dưới đối thủ FSHA chủ động ở hai mức grad_scale: gs=5.0 và gs=1.0.
# - Client: Nạp từ checkpoint đã huấn luyện (artape_v0_k1024_lam0.1_s42_client.pt)
# - Chỉ số ghi nhận: Acc hội tụ, PSNR và SSIM ở đỉnh (peak) và ở cuối (final epoch 30),
#   cùng LPIPS và ảnh tái tạo trực quan.
# ==============================================================================
import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from src.data.cifar import CIFAR10_MEAN, CIFAR10_STD
from src.data import get_cifar10_fsha_splits
from src.attacks.fsha import (
    FSHADiscriminator,
    FSHAPilotAutoEncoder,
    fit_fsha_pilot,
    warmup_fsha_discriminator,
)
from src.metrics import psnr_ssim, get_lpips_fn, calculate_lpips, distance_correlation
from src.metrics.reconstruction import denormalize
from src.utils.visualize import save_reconstruction_grid, plot_attack_curves

from train_artape_v0 import ClientWithBottleneck, ServerWithBottleneck, LearnedAdaptiveDecoder


class ARTAPEv0FSHADefense(nn.Module):
    """
    Toán tử phòng thủ AR-TAPE v0 tại cut layer:
    - Client bottleneck: Chiếu z in R^{B x 65536} -> z_t in R^{B x k} (k=1024)
    - Server unprojector: Nâng z_t in R^{B x k} -> z_hat in R^{B x 65536} -> (B, 64, 32, 32)
    Gradient đối kháng từ Discriminator truyền qua unprojector và bottleneck về Client.
    """
    def __init__(self, bottleneck_layer, unprojector_layer):
        super().__init__()
        self.bottleneck = bottleneck_layer
        self.unprojector = unprojector_layer

    def forward(self, z):
        b = z.size(0)
        z_flat = z.view(b, -1)
        zt = self.bottleneck(z_flat)
        zh = self.unprojector(zt)
        return zh.view(b, 64, 32, 32)


class FSHADecoderView(nn.Module):
    """
    Wrapper kết nối Pilot Decoder để xuất lưới ảnh tái tạo f^-1(Client(x))
    """
    def __init__(self, pilot_ae, defense_module):
        super().__init__()
        self.pilot_ae = pilot_ae
        self.defense = defense_module

    def forward(self, z):
        z_hat = self.defense(z)
        return self.pilot_ae.decode(z_hat)


def evaluate_fsha_step(client, defense, pilot_ae, task_server, loader, device, lpips_fn=None):
    client.eval()
    defense.eval()
    pilot_ae.eval()
    task_server.eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    lpips_sum = 0.0
    corr_task = 0
    total_samples = 0
    dcor_sum = 0.0
    n_dcor_b = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            b = x.size(0)

            # Client forward
            h = client.base_client(x)
            z_hat = defense(h)

            # Pilot reconstruction
            x_rec = pilot_ae.decode(z_hat)

            # Task prediction qua task_server
            logits = task_server(z_hat)
            corr_task += (logits.argmax(1) == y).sum().item()

            p, s = psnr_ssim(x, x_rec, CIFAR10_MEAN, CIFAR10_STD)
            psnr_sum += p * b
            ssim_sum += s * b
            mse_sum += F.mse_loss(x_rec, x).item() * b

            if lpips_fn is not None:
                lp = calculate_lpips(x, x_rec, CIFAR10_MEAN, CIFAR10_STD, lpips_fn)
                if lp is not None:
                    lpips_sum += lp * b

            if n_dcor_b < 5:
                dcor_sum += distance_correlation(x, z_hat).item()
                n_dcor_b += 1

            total_samples += b

    avg_psnr = psnr_sum / max(total_samples, 1)
    avg_ssim = ssim_sum / max(total_samples, 1)
    avg_mse = mse_sum / max(total_samples, 1)
    avg_lpips = (lpips_sum / max(total_samples, 1)) if lpips_fn is not None else None
    task_acc = corr_task / max(total_samples, 1)
    avg_dcor = dcor_sum / max(n_dcor_b, 1)

    return {
        "psnr": round(avg_psnr, 2),
        "ssim": round(avg_ssim, 4),
        "mse": round(avg_mse, 5),
        "lpips": round(avg_lpips, 4) if avg_lpips is not None else None,
        "task_acc": round(task_acc * 100, 2),
        "dcor": round(avg_dcor, 4)
    }


def run_artape_fsha_single(
    grad_scale,
    client,
    defense_mod,
    task_server,
    pilot_ae,
    priv_loader,
    pub_loader,
    test_eval_loader,
    test_full_loader,
    args,
    device,
    lpips_fn=None
):
    scenario_id = f"fsha_artape_v0_k{args.k}_lam{args.lambda_val}_s{args.seed}_gs{grad_scale}"
    print("\n" + "=" * 90)
    print(f"KHỞI CHẠY FSHA TRÊN AR-TAPE v0 | GRAD SCALE: {grad_scale}x | SEED: {args.seed}")
    print(f"Scenario ID: {scenario_id}")
    print("=" * 90, flush=True)

    # 1. Warmup Discriminator
    discriminator = FSHADiscriminator(in_channels=64).to(device)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr, betas=(0.5, 0.999))
    opt_c = torch.optim.SGD(client.parameters(), lr=args.client_lr, momentum=0.9, weight_decay=1e-4)
    sched_c = CosineAnnealingLR(opt_c, T_max=args.epochs)
    opt_task_s = torch.optim.SGD(
        list(task_server.parameters()) + list(defense_mod.unprojector.parameters()),
        lr=0.005,
        momentum=0.9,
        weight_decay=1e-4
    )

    print("[FSHA] Bắt đầu khởi động Discriminator (warmup)...", flush=True)
    warmup_fsha_discriminator(
        client=client.base_client,
        discriminator=discriminator,
        pilot_ae=pilot_ae,
        priv_loader=priv_loader,
        pub_loader=pub_loader,
        opt_d=opt_d,
        defense=defense_mod,
        device=device,
        num_batches=20
    )

    # 2. Đo đạc trước khi bị Hijack (Epoch 0)
    init_eval = evaluate_fsha_step(client, defense_mod, pilot_ae, task_server, test_eval_loader, device, lpips_fn)
    print(f"[Trước Hijack (Ep 0)] Task Acc: {init_eval['task_acc']:.2f}% | PSNR: {init_eval['psnr']:.2f} dB | SSIM: {init_eval['ssim']:.4f} | dCor: {init_eval['dcor']:.4f}", flush=True)

    history = []
    best_psnr = -1.0
    best_ssim = -1.0
    best_epoch = 0
    t_start = time.time()

    criterion_ce = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        client.train()
        defense_mod.train()
        task_server.train()
        discriminator.train()

        pub_iter = iter(pub_loader)
        total_loss_d = 0.0
        total_loss_adv = 0.0
        n_batches = 0

        for x_priv, y_priv in priv_loader:
            x_priv, y_priv = x_priv.to(device, non_blocking=True), y_priv.to(device, non_blocking=True)
            try:
                x_pub, _ = next(pub_iter)
            except StopIteration:
                pub_iter = iter(pub_loader)
                x_pub, _ = next(pub_iter)
            x_pub = x_pub.to(device, non_blocking=True)

            # ---------------------------------------------------------
            # BƯỚC A: Cập nhật Discriminator
            # ---------------------------------------------------------
            opt_d.zero_grad()
            with torch.no_grad():
                h_priv = client.base_client(x_priv)
                z_priv = defense_mod(h_priv)
                z_pub, _ = pilot_ae(x_pub)

            score_priv = discriminator(z_priv)
            score_pub = discriminator(z_pub)

            # WGAN critic loss / Relativistic loss
            loss_d = (score_priv.mean() - score_pub.mean())
            loss_d.backward()
            opt_d.step()
            total_loss_d += loss_d.item()

            # ---------------------------------------------------------
            # BƯỚC B: Hijacking Client qua Gradient độc hại từ Server
            # ---------------------------------------------------------
            opt_c.zero_grad()
            opt_task_s.zero_grad()

            h_priv_c = client.base_client(x_priv)
            z_priv_c = defense_mod(h_priv_c)

            # 1. Task loss hợp pháp từ Server
            logits = task_server(z_priv_c)
            loss_task = criterion_ce(logits, y_priv)

            # 2. Adversarial loss (ép Client hướng về Pilot space)
            score_hijack = discriminator(z_priv_c)
            loss_adv = - score_hijack.mean()
            total_loss_adv += loss_adv.item()

            # Tổng hợp gradient gửi về Client: g = grad_scale * loss_adv + loss_task
            loss_hijack = loss_task + grad_scale * loss_adv
            loss_hijack.backward()

            opt_c.step()
            opt_task_s.step()
            n_batches += 1

        sched_c.step()
        ep_time = time.time() - t_ep

        # Đánh giá sau mỗi epoch
        eval_res = evaluate_fsha_step(client, defense_mod, pilot_ae, task_server, test_eval_loader, device, lpips_fn)

        if eval_res["psnr"] > best_psnr:
            best_psnr = eval_res["psnr"]
            best_ssim = eval_res["ssim"]
            best_epoch = ep

        eval_res["epoch"] = ep
        eval_res["loss_d"] = round(total_loss_d / max(n_batches, 1), 4)
        eval_res["loss_adv"] = round(total_loss_adv / max(n_batches, 1), 4)
        eval_res["epoch_time_s"] = round(ep_time, 1)
        history.append(eval_res)

        print(f"  [FSHA Ep {ep:2d}/{args.epochs}] ({ep_time:.1f}s) | Task Acc: {eval_res['task_acc']:.2f}% | "
              f"PSNR: {eval_res['psnr']:.2f} dB (Peak: {best_psnr:.2f}dB Ep {best_epoch}) | "
              f"SSIM: {eval_res['ssim']:.4f} | LossD: {eval_res['loss_d']:.4f} | dCor: {eval_res['dcor']:.4f}",
              flush=True)

    total_time = time.time() - t_start

    # Đánh giá cuối cùng trên Full 10.000 test
    full_test_res = evaluate_fsha_step(client, defense_mod, pilot_ae, task_server, test_full_loader, device, lpips_fn)

    # Xuất ảnh trực quan hóa
    recon_view = FSHADecoderView(pilot_ae, defense_mod).to(device)
    grid_path = os.path.join(args.output_dir, f"fsha_recon_{scenario_id}.png")
    save_reconstruction_grid(client.base_client, recon_view, test_eval_loader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=grid_path, num_images=8)

    # Biểu đồ hội tụ
    plot_fsha_curves(history, os.path.join(args.output_dir, f"fsha_curves_{scenario_id}.png"))

    final_psnr = eval_res["psnr"]
    final_ssim = eval_res["ssim"]
    final_task_acc = full_test_res["task_acc"]

    is_broken = (best_psnr >= 18.0 or best_ssim >= 0.50)
    if is_broken:
        verdict = "Sụp đổ trước FSHA (Đối thủ tái tạo thành công)"
    elif final_task_acc < 70.0:
        verdict = "Chặn được tái tạo nhưng Utility sụp đổ nặng"
    elif final_task_acc < 85.0:
        verdict = "Chặn được tái tạo nhưng Utility suy giảm nhẹ"
    else:
        verdict = "Kháng được FSHA & Giữ vững Utility"

    record = {
        "scenario_id": scenario_id,
        "defense": "AR_TAPE_v0",
        "k": args.k,
        "lambda": args.lambda_val,
        "seed": args.seed,
        "grad_scale": grad_scale,
        "converged_task_acc_full": round(final_task_acc, 2),
        "converged_task_acc_2k": eval_res["task_acc"],
        "peak_psnr": round(best_psnr, 2),
        "peak_ssim": round(best_ssim, 4),
        "peak_epoch": best_epoch,
        "final_psnr": round(final_psnr, 2),
        "final_ssim": round(final_ssim, 4),
        "final_lpips": eval_res["lpips"],
        "final_dcor": eval_res["dcor"],
        "init_psnr": init_eval["psnr"],
        "init_ssim": init_eval["ssim"],
        "total_time_s": round(total_time, 1),
        "verdict": verdict,
        "history": history
    }

    print("\n" + "-" * 85)
    print(f"KẾT QUẢ FSHA gs={grad_scale} TRÊN AR-TAPE v0:")
    print(f"  - Acc Hội Tụ (Full 10k Test): {final_task_acc:.2f}%")
    print(f"  - PSNR ở Đỉnh: {best_psnr:.2f} dB (Epoch {best_epoch}) | PSNR ở Cuối: {final_psnr:.2f} dB")
    print(f"  - SSIM ở Đỉnh: {best_ssim:.4f} (Epoch {best_epoch}) | SSIM ở Cuối: {final_ssim:.4f}")
    print(f"  - Nhận định: {verdict}")
    print("-" * 85 + "\n", flush=True)

    return record


def plot_fsha_curves(history, save_path):
    epochs = [h["epoch"] for h in history]
    psnrs = [h["psnr"] for h in history]
    ssims = [h["ssim"] for h in history]
    accs = [h["task_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, psnrs, "b-o", label="Reconstruction PSNR (dB)")
    ax1.axhline(15.0, color="green", linestyle="--", label="Ngưỡng Privacy (15 dB)")
    ax1.axhline(20.0, color="red", linestyle=":", label="Ngưỡng Sụp Đổ (20 dB)")
    ax1.set_xlabel("Epoch FSHA")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Chất Lượng Tái Tạo FSHA theo Epoch")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, accs, "r-s", label="Task Test Acc (%)")
    ax2.plot(epochs, [s * 100 for s in ssims], "g--^", label="SSIM x100")
    ax2.axhline(92.7, color="gray", linestyle=":", label="Baseline Acc (92.7%)")
    ax2.set_xlabel("Epoch FSHA")
    ax2.set_ylabel("Độ chính xác / SSIM x100")
    ax2.set_title("Độ Ổn Định Nhiệm Vụ (Utility) và SSIM")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Chạy FSHA gs=5 và gs=1 trên Client AR-TAPE v0")
    parser.add_argument("--client-ckpt", type=str, default=None, help="Đường dẫn checkpoint client AR-TAPE v0")
    parser.add_argument("--server-ckpt", type=str, default=None, help="Đường dẫn checkpoint server AR-TAPE v0")
    parser.add_argument("--k", type=int, default=1024, help="Số chiều bottleneck k (mặc định: 1024)")
    parser.add_argument("--m", type=int, default=512, help="Số chiều ẩn MLP unprojector (mặc định: 512)")
    parser.add_argument("--target-dim", type=int, default=32, help="Chiều không gian mục tiêu d_target của Pilot AE (mặc định: 32)")
    parser.add_argument("--lambda-val", type=float, default=0.1, help="Giá trị lambda của checkpoint (mặc định: 0.1)")
    parser.add_argument("--grad-scales", nargs="+", type=float, default=[5.0, 1.0], help="Danh sách grad scale (mặc định: 5.0 1.0)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs tấn công FSHA (mặc định: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (mặc định: 42)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--client-lr", type=float, default=1e-3, help="Learning rate cập nhật Client (mặc định: 1e-3)")
    parser.add_argument("--disc-lr", type=float, default=2e-4, help="Learning rate Discriminator (mặc định: 2e-4)")
    parser.add_argument("--warmup-d-epochs", type=int, default=3, help="Số epoch warmup discriminator (mặc định: 3)")
    parser.add_argument("--pilot-epochs", type=int, default=25, help="Số epoch fit pilot autoencoder nếu chưa có (mặc định: 25)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_v0_FSHA"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Tìm kiếm checkpoint client mặc định
    if args.client_ckpt is None or not os.path.isfile(args.client_ckpt):
        candidates = [
            os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v0", "checkpoints", f"artape_v0_k{args.k}_lam{args.lambda_val}_s{args.seed}_client.pt"),
            os.path.join(PROJECT_ROOT, "output", "AbReTAPE_v0", "checkpoints", f"artape_v0_k{args.k}_lam{args.lambda_val}_s{args.seed}_client.pt"),
            f"/content/drive/MyDrive/AbReTAPE_Results/AbReTAPE_v0/checkpoints/artape_v0_k{args.k}_lam{args.lambda_val}_s{args.seed}_client.pt",
            f"/content/drive/MyDrive/AbReTAPE_Results/AR_TAPE_v0/checkpoints/artape_v0_k{args.k}_lam{args.lambda_val}_s{args.seed}_client.pt",
        ]
        for c in candidates:
            if os.path.isfile(c):
                args.client_ckpt = c
                break

    if args.server_ckpt is None or not os.path.isfile(args.server_ckpt):
        s_candidates = [
            os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v0", "checkpoints", f"artape_v0_k{args.k}_lam{args.lambda_val}_s{args.seed}_server.pt"),
            os.path.join(PROJECT_ROOT, "output", "AbReTAPE_v0", "checkpoints", f"artape_v0_k{args.k}_lam{args.lambda_val}_s{args.seed}_server.pt"),
            f"/content/drive/MyDrive/AbReTAPE_Results/AbReTAPE_v0/checkpoints/artape_v0_k{args.k}_lam{args.lambda_val}_s{args.seed}_server.pt",
            f"/content/drive/MyDrive/AbReTAPE_Results/AR_TAPE_v0/checkpoints/artape_v0_k{args.k}_lam{args.lambda_val}_s{args.seed}_server.pt",
        ]
        for sc in s_candidates:
            if os.path.isfile(sc):
                args.server_ckpt = sc
                break

    if args.client_ckpt is None or not os.path.isfile(args.client_ckpt):
        print(f"[LỖI] Không tìm thấy Client Checkpoint AR-TAPE v0 tại: {args.client_ckpt}")
        print("Vui lòng chỉ định rõ qua cờ --client-ckpt hoặc đồng bộ từ Google Drive.")
        sys.exit(1)

    print("=" * 90)
    print("THỰC NGHIỆM ĐÁNH GIÁ TẤN CÔNG CHỦ ĐỘNG FSHA TRÊN CLIENT AR-TAPE v0")
    print(f"Thiết bị: {device} | Seed: {args.seed} | Bottleneck k={args.k}")
    print(f"Client Checkpoint: {args.client_ckpt}")
    print(f"Danh sách Grad Scales: {args.grad_scales} | Epochs: {args.epochs}")
    print(f"Thư mục lưu trữ: {args.output_dir}")
    print("=" * 90, flush=True)

    # 1. Nạp dữ liệu FSHA splits
    priv_loader, pub_loader, test_full_loader = get_cifar10_fsha_splits(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0 if sys.platform == "win32" else 2,
        seed=args.seed
    )

    # Tạo tập đánh giá nhanh test_eval_loader (2000 ảnh) từ test set
    g_eval = torch.Generator().manual_seed(args.seed)
    eval_indices = torch.randperm(len(test_full_loader.dataset), generator=g_eval)[:2000].tolist()
    test_eval_set = torch.utils.data.Subset(test_full_loader.dataset, eval_indices)
    test_eval_loader = torch.utils.data.DataLoader(
        test_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=0 if sys.platform == "win32" else 2
    )

    # 2. Chuẩn bị Pilot AutoEncoder (f_tilde, f_tilde^-1)
    pilot_ae_path = os.path.join(args.output_dir, f"pilot_ae_cifar10_s{args.seed}.pt")
    target_dim = args.target_dim if args.target_dim > 0 else None
    pilot_ae = FSHAPilotAutoEncoder(in_channels=3, latent_channels=64, target_dim=target_dim).to(device)

    # Tái sử dụng pilot fitted nếu đã có
    existing_pilots = [
        pilot_ae_path,
        os.path.join(PROJECT_ROOT, "output", "AbReTAPE_FSHA", "pilot_ae_fitted_d32.pt"),
        "/content/drive/MyDrive/AbReTAPE_Results/AbReTAPE_FSHA/pilot_ae_fitted_d32.pt",
        "/content/drive/MyDrive/AbReTAPE_Results/AbReTAPE_FSHA/pilot_ae_cifar10_s42.pt",
    ]
    pilot_loaded = False
    for p_cand in existing_pilots:
        if os.path.isfile(p_cand):
            try:
                ckpt = torch.load(p_cand, map_location=device)
                if isinstance(ckpt, dict) and "pilot_ae" in ckpt:
                    pilot_ae.load_state_dict(ckpt["pilot_ae"], strict=False)
                else:
                    pilot_ae.load_state_dict(ckpt, strict=False)
                print(f"[PILOT] Đã nạp Pilot AutoEncoder có sẵn từ: {p_cand}")
                pilot_loaded = True
                break
            except Exception as e:
                print(f"[PILOT] Bỏ qua {p_cand}: {e}")

    if pilot_loaded:
        pilot_ae.eval()
        for p in pilot_ae.parameters():
            p.requires_grad = False
    else:
        print(f"[PILOT] Đang huấn luyện Pilot AutoEncoder trên tập công khai ({args.pilot_epochs} epochs)...", flush=True)
        fit_pilot_server = ServerModel().to(device)
        fit_fsha_pilot(
            pilot_ae=pilot_ae,
            pilot_server=fit_pilot_server,
            pub_loader=pub_loader,
            test_loader=test_eval_loader,
            epochs=args.pilot_epochs,
            lr=1e-3,
            device=device,
            save_path=pilot_ae_path
        )
        print(f"[PILOT] Đã lưu Pilot AutoEncoder tại: {pilot_ae_path}")

    lpips_fn = get_lpips_fn(device=device)

    all_records = []
    for gs in args.grad_scales:
        # Nạp Client mới từ checkpoint cho mỗi kịch bản grad_scale
        client = ClientWithBottleneck(k=args.k, D=65536, init_ortho=False).to(device)
        client.load_state_dict(torch.load(args.client_ckpt, map_location=device))

        # Khởi tạo Server tương ứng
        server = ServerWithBottleneck(k=args.k, m=args.m, D=65536).to(device)
        if args.server_ckpt and os.path.isfile(args.server_ckpt):
            try:
                server.load_state_dict(torch.load(args.server_ckpt, map_location=device))
                print(f"[SERVER] Đã nạp Server Checkpoint từ {args.server_ckpt}")
            except Exception:
                pass

        # Tạo module phòng thủ AR-TAPE v0
        defense_mod = ARTAPEv0FSHADefense(client.bottleneck, server.unprojector).to(device)
        task_server = server.base_server

        rec = run_artape_fsha_single(
            grad_scale=gs,
            client=client,
            defense_mod=defense_mod,
            task_server=task_server,
            pilot_ae=pilot_ae,
            priv_loader=priv_loader,
            pub_loader=pub_loader,
            test_eval_loader=test_eval_loader,
            test_full_loader=test_full_loader,
            args=args,
            device=device,
            lpips_fn=lpips_fn
        )
        all_records.append(rec)

    # Lưu kết quả tổng hợp JSON & CSV
    json_path = os.path.join(args.output_dir, "results_artape_v0_fsha.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output_dir, "results_artape_v0_fsha.csv")
    fieldnames = [
        "scenario_id", "defense", "k", "lambda", "seed", "grad_scale",
        "converged_task_acc_full", "converged_task_acc_2k",
        "peak_psnr", "peak_ssim", "peak_epoch", "final_psnr", "final_ssim",
        "final_lpips", "final_dcor", "init_psnr", "init_ssim", "total_time_s", "verdict"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 90)
    print("BẢNG TỔNG HỢP KẾT QUẢ TẤN CÔNG FSHA TRÊN CLIENT AR-TAPE v0:")
    print(f"{'Grad Scale':<12} | {'Acc Hội Tụ (10k)':<18} | {'Peak PSNR':<11} | {'Peak SSIM':<11} | {'Final PSNR':<11} | {'Final SSIM':<11} | {'Nhận Định'}")
    print("-" * 90)
    for r in all_records:
        print(f"gs={r['grad_scale']:<9} | {r['converged_task_acc_full']:<18.2f}% | {r['peak_psnr']:<11.2f} | {r['peak_ssim']:<11.4f} | {r['final_psnr']:<11.2f} | {r['final_ssim']:<11.4f} | {r['verdict']}")
    print("=" * 90)
    print(f"Đã lưu kết quả tại: {json_path} và {csv_path}")


if __name__ == "__main__":
    main()
