#!/usr/bin/env python3
# Bước 3: Huấn luyện và Đánh giá các Baseline phòng thủ kinh điển: B1 (Gaussian Noise) & B2 (DP-SGD)
import os
import sys
import time
import json
import csv
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel, ServerModel
from src.data import get_cifar10, CIFAR10_MEAN, CIFAR10_STD
from src.defenses import GaussianNoise, DPSGDClientOptimizer, compute_dp_epsilon
from src.training import train_sl_epoch, evaluate_sl
from src.attacks import Decoder, train_inversion_epoch, evaluate_inversion
from src.metrics import get_lpips_fn
from src.utils import plot_tradeoff_curves, save_reconstruction_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 3: Đánh giá Baseline B1 (Gaussian Noise) và B2 (DP-SGD)")
    parser.add_argument("--defense", choices=["b1", "b2", "all"], default="b1", help="Lựa chọn baseline: 'b1' (Gaussian Noise), 'b2' (DP-SGD), hoặc 'all'")
    parser.add_argument("--epochs", type=int, default=10, help="Số epochs huấn luyện SL (mặc định: 10)")
    parser.add_argument("--attack-epochs", type=int, default=10, help="Số epochs huấn luyện Decoder tấn công (mặc định: 10)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate cho SL (mặc định: 0.05)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục dữ liệu")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output"), help="Thư mục lưu outputs")
    return parser.parse_args()


def run_b1_gaussian(args, device):
    out_dir = os.path.join(args.output_dir, "AbReTAPE_Step3")
    os.makedirs(out_dir, exist_ok=True)
    sigmas = [0.1, 0.5, 1.0]
    results = []

    print("\n" + "=" * 70)
    print("BƯỚC 3 — BASELINE B1: GAUSSIAN NOISE AT CUT LAYER")
    print(f"Thử nghiệm với các mức sigma: {sigmas}")
    print("=" * 70)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size)
    lpips_fn = get_lpips_fn(device=device)

    for sigma in sigmas:
        print(f"\n---> ĐANG CHẠY BASELINE B1 VỚI SIGMA = {sigma} <---")
        client = ClientModel().to(device)
        server = ServerModel().to(device)
        noise = GaussianNoise(sigma=sigma).to(device)

        opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        # 1. Huấn luyện Split Learning với nhiễu
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_sl_epoch(client, server, trainloader, opt_c, opt_s, criterion, device, defense=noise)
            print(f"[SL Train σ={sigma}] Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%", flush=True)

        _, test_acc = evaluate_sl(client, server, testloader, device, defense=noise, criterion=criterion)
        print(f"[SL Result σ={sigma}] Final Test Accuracy: {test_acc*100:.2f}%")

        # 2. Đánh giá tấn công Feature Inversion Attack
        print(f"[Attack] Đang huấn luyện Decoder tấn công đối phó với σ = {sigma}...")
        decoder = Decoder().to(device)
        opt_d = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-5)
        crit_d = nn.MSELoss()

        for ep in range(1, args.attack_epochs + 1):
            train_inversion_epoch(client, decoder, trainloader, opt_d, crit_d, device, defense=noise)

        mse, psnr, ssim, lpips_val = evaluate_inversion(
            client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, defense=noise, criterion=crit_d, lpips_fn=lpips_fn
        )
        print(f"[Security σ={sigma}] PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | LPIPS: {lpips_val if lpips_val else 'N/A'}")

        results.append({
            "sigma": sigma,
            "test_acc": test_acc,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_val
        })

    # Lưu kết quả
    res_file = os.path.join(out_dir, "results_b1.json")
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_file = os.path.join(out_dir, "b1_tradeoff_curves.png")
    plot_tradeoff_curves(results, save_path=plot_file, x_key="sigma", x_label="Cường độ Nhiễu Gauss (sigma)")
    print(f"\n[DONE] Hoàn thành Baseline B1! Kết quả lưu tại: {res_file}")


def run_b2_dpsgd(args, device):
    out_dir = os.path.join(args.output_dir, "AbReTAPE_Step3_B2")
    os.makedirs(out_dir, exist_ok=True)
    sigmas_dp = [0.5, 1.0, 2.0]
    results = []

    print("\n" + "=" * 70)
    print("BƯỚC 3 — BASELINE B2: DP-SGD CLIENT OPTIMIZER")
    print(f"Thử nghiệm với các mức sigma_DP: {sigmas_dp}")
    print("=" * 70)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size)
    lpips_fn = get_lpips_fn(device=device)

    for s_dp in sigmas_dp:
        eps = compute_dp_epsilon(epochs=args.epochs, batch_size=args.batch_size, noise_multiplier=s_dp)
        print(f"\n---> ĐANG CHẠY BASELINE B2: σ_DP = {s_dp} (Epsilon: {eps:.2f}, Delta: 1e-5) <---")

        client = ClientModel().to(device)
        server = ServerModel().to(device)

        base_opt = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        dp_opt = DPSGDClientOptimizer(client, base_opt, max_grad_norm=1.0, noise_multiplier=s_dp)
        opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_sl_epoch(client, server, trainloader, dp_opt, opt_s, criterion, device, client_opt_is_dpsgd=True)
            print(f"[DP-SL σ={s_dp}] Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%", flush=True)

        _, test_acc = evaluate_sl(client, server, testloader, device, criterion=criterion)
        print(f"[DP-SL Result σ={s_dp}] Final Test Accuracy: {test_acc*100:.2f}% (Epsilon: {eps:.2f})")

        # Đánh giá Feature Inversion Attack
        decoder = Decoder().to(device)
        opt_d = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-5)
        crit_d = nn.MSELoss()

        for ep in range(1, args.attack_epochs + 1):
            train_inversion_epoch(client, decoder, trainloader, opt_d, crit_d, device)

        mse, psnr, ssim, lpips_val = evaluate_inversion(
            client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, criterion=crit_d, lpips_fn=lpips_fn
        )
        print(f"[Security DP σ={s_dp}] PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

        results.append({
            "sigma_dp": s_dp,
            "epsilon": eps,
            "test_acc": test_acc,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_val
        })

    res_file = os.path.join(out_dir, "results_b2.json")
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_file = os.path.join(out_dir, "b2_tradeoff_curves.png")
    plot_tradeoff_curves(results, save_path=plot_file, x_key="sigma_dp", x_label="Cường độ Nhiễu DP (sigma_DP)")
    print(f"\n[DONE] Hoàn thành Baseline B2! Kết quả lưu tại: {res_file}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.defense in ["b1", "all"]:
        run_b1_gaussian(args, device)
    if args.defense in ["b2", "all"]:
        run_b2_dpsgd(args, device)


if __name__ == "__main__":
    main()
