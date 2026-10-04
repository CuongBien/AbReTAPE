#!/usr/bin/env python3
# ==============================================================================
# BƯỚC THỰC NGHIỆM: ATTACKER CO-ADAPTED THỤ ĐỘNG (BẢN NHẸ ONLINE SHADOW INVERSION)
# ==============================================================================
# Đánh giá cận trên năng lực tấn công thụ động (Honest-but-curious) qua 3 nhánh đối chứng:
# - Nhánh A: Co-adapted (Online): Decoder cập nhật song song mỗi epoch cùng Split Learning trên D_aux
# - Nhánh B: Fresh Matched Steps: Decoder huấn luyện trên client đóng băng với đúng tổng số bước của Nhánh A
# - Nhánh C: Fresh 30 Epochs: Giao thức chuẩn 30 epochs
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.models import ClientModel, ServerModel
from src.data import CIFAR10_MEAN, CIFAR10_STD
from src.defenses import BlockScrambleDefense, FixedOrthoProjection
from src.attacks import (
    Decoder,
    get_coadapted_data_splits,
    CoAdaptedPassiveAttacker,
    evaluate_coadapted_inversion,
    train_fresh_matched_steps,
    train_fresh_epochs,
)
from src.metrics import get_lpips_fn, distance_correlation
from src.metrics.reconstruction import denormalize
from src.training.split_learning import evaluate_sl


def parse_args():
    parser = argparse.ArgumentParser(description="Attacker Co-adapted Thụ Động (Online Shadow Inversion)")
    parser.add_argument("--defenses", type=str, default="b0,b4",
                        help="Danh sách baseline cần chạy, ngăn cách bởi dấu phẩy (vd: 'b0,b4' hoặc 'b0,b4,b7')")
    parser.add_argument("--seeds", type=str, default="42,7",
                        help="Danh sách seed ngẫu nhiên (vd: '42,7')")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Số epochs huấn luyện Split Learning (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate cho SGD của SL (mặc định: 0.1)")
    parser.add_argument("--decoder-lr", type=float, default=1e-3,
                        help="Learning rate cho Adam của Decoder (mặc định: 1e-3)")
    parser.add_argument("--fresh-30ep", type=int, default=30,
                        help="Số epochs cho Nhánh C Fresh chuẩn (mặc định: 30)")
    parser.add_argument("--aux-size", type=int, default=2500,
                        help="Kích thước tập ảnh phụ cố định D_aux (mặc định: 2500)")
    parser.add_argument("--test-eval-size", type=int, default=2000,
                        help="Kích thước tập con test cố định để đánh giá nhanh (mặc định: 2000)")
    parser.add_argument("--eval-freq", type=int, default=5,
                        help="Tần suất đánh giá mỗi N epochs (mặc định: 5)")
    parser.add_argument("--b4-block-size", type=int, default=4,
                        help="Kích thước block cho B4 Block Scramble (mặc định: 4)")
    parser.add_argument("--b7-k", type=int, default=1024,
                        help="Số chiều chiếu k cho B7 LightSplit (mặc định: 1024)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2,
                        help="Số workers nạp dữ liệu")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"),
                        help="Thư mục dữ liệu CIFAR-10")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_CoAdapted"),
                        help="Thư mục xuất kết quả")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Thư mục sao lưu Google Drive (nếu chạy Colab)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Bỏ qua các kịch bản đã có kết quả trong JSON")
    return parser.parse_args()


@torch.no_grad()
def save_3branch_comparison_grid(
    client, defense, dec_coadapted, dec_matched, dec_30ep,
    loader, device, save_path, num_images=6
):
    """
    Xuất lưới ảnh đối chứng 4 hàng x num_images cột:
    Hàng 1: Ảnh gốc x
    Hàng 2: Nhánh A (Co-adapted Online)
    Hàng 3: Nhánh B (Fresh Matched Steps)
    Hàng 4: Nhánh C (Fresh 30 Epochs)
    """
    client.eval()
    if defense is not None:
        defense.eval()
    dec_coadapted.eval()
    dec_matched.eval()
    dec_30ep.eval()

    imgs, targets = [], []
    for x, y in loader:
        imgs.append(x)
        targets.append(y)
        if sum(t.size(0) for t in targets) >= num_images:
            break

    x_all = torch.cat(imgs, dim=0)[:num_images].to(device)

    z = client(x_all)
    if defense is not None:
        z = defense(z)

    x_coadapt = dec_coadapted(z)
    x_matched = dec_matched(z)
    x_30ep = dec_30ep(z)

    mean, std = CIFAR10_MEAN, CIFAR10_STD
    x_orig_np = denormalize(x_all, mean, std).cpu().numpy()
    x_coadapt_np = denormalize(x_coadapt, mean, std).cpu().numpy()
    x_matched_np = denormalize(x_matched, mean, std).cpu().numpy()
    x_30ep_np = denormalize(x_30ep, mean, std).cpu().numpy()

    fig, axes = plt.subplots(4, num_images, figsize=(num_images * 2.3, 9.2))
    row_labels = [
        "Ảnh Gốc\nx",
        "Nhánh A\n(Co-adapted)",
        "Nhánh B\n(Fresh-Matched)",
        "Nhánh C\n(Fresh-30ep)"
    ]

    for i in range(num_images):
        orig = x_orig_np[i].transpose(1, 2, 0)
        coad = x_coadapt_np[i].transpose(1, 2, 0)
        matc = x_matched_np[i].transpose(1, 2, 0)
        ep30 = x_30ep_np[i].transpose(1, 2, 0)

        # Hàng 1
        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Mẫu #{i+1}", fontsize=11, fontweight="bold")

        # Hàng 2: Co-adapted
        p_c = peak_signal_noise_ratio(orig, coad, data_range=1.0)
        s_c = structural_similarity(orig, coad, channel_axis=2, data_range=1.0)
        axes[1, i].imshow(coad)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"{p_c:.1f}dB | {s_c:.2f}", fontsize=10, color="#d62728" if s_c > 0.4 else "#2ca02c")

        # Hàng 3: Matched
        p_m = peak_signal_noise_ratio(orig, matc, data_range=1.0)
        s_m = structural_similarity(orig, matc, channel_axis=2, data_range=1.0)
        axes[2, i].imshow(matc)
        axes[2, i].axis("off")
        axes[2, i].set_title(f"{p_m:.1f}dB | {s_m:.2f}", fontsize=10, color="#d62728" if s_m > 0.4 else "#2ca02c")

        # Hàng 4: 30ep
        p_3 = peak_signal_noise_ratio(orig, ep30, data_range=1.0)
        s_3 = structural_similarity(orig, ep30, channel_axis=2, data_range=1.0)
        axes[3, i].imshow(ep30)
        axes[3, i].axis("off")
        axes[3, i].set_title(f"{p_3:.1f}dB | {s_3:.2f}", fontsize=10, color="#d62728" if s_3 > 0.4 else "#2ca02c")

    for r in range(4):
        axes[r, 0].text(-0.25, 0.5, row_labels[r], transform=axes[r, 0].transAxes,
                        fontsize=11, fontweight="bold", va="center", ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[GRID] Đã xuất lưới ảnh đối chứng 3 nhánh tại: {save_path}", flush=True)


def plot_coadapted_curves(history, save_path):
    """
    Vẽ đồ thị tiến trình Co-adaptation (Nhánh A) qua các epochs:
    (a) Test Accuracy của SL
    (b) PSNR của Co-adapted Decoder
    (c) SSIM của Co-adapted Decoder
    """
    epochs = [h["epoch"] for h in history]
    accs = [h["test_acc"] * 100 for h in history]
    psnrs = [h["psnr"] for h in history]
    ssims = [h["ssim"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Acc
    axes[0].plot(epochs, accs, marker="o", color="#1f77b4", linewidth=2.0)
    axes[0].set_title("Độ chính xác Phân loại SL (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # PSNR
    axes[1].plot(epochs, psnrs, marker="s", color="#d62728", linewidth=2.0)
    axes[1].axhline(20.0, color="#b2182b", linestyle="--", label="Ngưỡng 20 dB")
    axes[1].set_title("PSNR Co-adapted Decoder (dB)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # SSIM
    axes[2].plot(epochs, ssims, marker="^", color="#9467bd", linewidth=2.0)
    axes[2].axhline(0.40, color="#b2182b", linestyle="--", label="Ngưỡng SSIM 0.40")
    axes[2].set_title("SSIM Co-adapted Decoder", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def run_single_scenario(def_name, seed, args, device, lpips_fn):
    scenario_id = f"{def_name}_seed{seed}"
    print("\n" + "=" * 90)
    print(f"BẮT ĐẦU KỊCH BẢN CO-ADAPTED: {scenario_id.upper()}")
    print(f"Thiết bị: {device} | Seed: {seed} | Epochs SL: {args.epochs} | D_aux: {args.aux_size} mẫu")
    print("=" * 90, flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Nạp data splits chuẩn
    sl_loader, aux_loader, test_eval_loader, test_full_loader = get_coadapted_data_splits(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        aux_size=args.aux_size,
        test_eval_size=args.test_eval_size,
        num_workers=args.num_workers,
        seed=seed,
    )

    # 2. Khởi tạo Client và Server
    client = ClientModel().to(device)
    server = ServerModel().to(device)

    # 3. Khởi tạo cơ chế phòng thủ
    defense = None
    decoder_ctor = lambda: Decoder(in_channels=64, out_channels=3)

    if def_name == "b0":
        defense = None
    elif def_name == "b4":
        defense = BlockScrambleDefense(block_size=args.b4_block_size, seed=seed).to(device)
    elif def_name == "b7":
        defense = FixedOrthoProjection(k=args.b7_k, seed=seed, mode="F", device=device).to(device)
    else:
        raise ValueError(f"Không hỗ trợ baseline: {def_name}")

    # 4. Khởi tạo Optimizers cho Split Learning
    opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched_c = CosineAnnealingLR(opt_c, T_max=args.epochs)
    sched_s = CosineAnnealingLR(opt_s, T_max=args.epochs)
    criterion_task = nn.CrossEntropyLoss()

    # 5. Khởi tạo Attacker Co-adapted Thụ Động (Nhánh A)
    co_attacker = CoAdaptedPassiveAttacker(
        decoder=decoder_ctor(), lr=args.decoder_lr, device=device
    )

    print(f"\n[BƯỚC 1] Huấn luyện Split Learning 100 Epochs đồng bộ Nhánh A (Co-adapted Online)...", flush=True)
    history_coadapt = []
    best_coadapt_psnr = 0.0
    best_coadapt_ssim = 0.0
    best_coadapt_ep = 0

    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        # 1. Huấn luyện 1 epoch Split Learning chuẩn (Giao thức trung thực)
        client.train()
        server.train()
        if defense is not None:
            defense.train()

        sl_loss_sum = 0.0
        sl_correct = 0
        sl_samples = 0

        for x, y in sl_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            b_size = x.size(0)

            opt_s.zero_grad()
            opt_c.zero_grad()

            z = client(x)
            if defense is not None:
                z_trans = defense(z)
                z_d = z_trans.detach().requires_grad_(True)
            else:
                z_d = z.detach().requires_grad_(True)

            logits = server(z_d)
            loss_task = criterion_task(logits, y)
            loss_task.backward()
            opt_s.step()

            # Gradient task chuẩn gửi về client
            grad_boundary = z_d.grad
            target_z = z_trans if defense is not None else z
            target_z.backward(grad_boundary)
            opt_c.step()

            sl_loss_sum += loss_task.item() * b_size
            sl_correct += (logits.argmax(1) == y).sum().item()
            sl_samples += b_size

        sched_c.step()
        sched_s.step()

        # 2. Nhánh A bước 1 epoch trên D_aux bám theo Client hiện tại
        co_loss = co_attacker.step_epoch(client=client, aux_loader=aux_loader, defense=defense)

        # 3. Đánh giá định kỳ
        if ep == 1 or ep % args.eval_freq == 0 or ep == args.epochs:
            # Task Test Acc
            test_loss, test_acc = evaluate_sl(client, server, test_eval_loader, device, defense=defense, criterion=criterion_task)

            # Inversion Metrics của Co-adapted Decoder
            mse_val, psnr_val, ssim_val, lpips_val = evaluate_coadapted_inversion(
                decoder=co_attacker.decoder,
                client=client,
                loader=test_eval_loader,
                device=device,
                defense=defense,
                lpips_fn=lpips_fn if ep == args.epochs else None,
            )

            if psnr_val > best_coadapt_psnr:
                best_coadapt_psnr = psnr_val
                best_coadapt_ssim = ssim_val
                best_coadapt_ep = ep

            history_coadapt.append({
                "epoch": ep,
                "sl_loss": round(sl_loss_sum / sl_samples, 4),
                "test_acc": round(test_acc, 4),
                "co_recon_loss": round(co_loss, 5),
                "psnr": round(psnr_val, 2),
                "ssim": round(ssim_val, 4),
                "lpips": round(lpips_val, 4) if lpips_val is not None else None,
            })

            print(
                f"  [Ep {ep:3d}/{args.epochs}] Task Acc: {test_acc*100:6.2f}% | "
                f"Co-adapt PSNR: {psnr_val:5.2f} dB | SSIM: {ssim_val:6.4f} (Peak: {best_coadapt_psnr:.2f}dB Ep {best_coadapt_ep})",
                flush=True
            )

    t_sl = time.time() - t_start
    total_coadapt_steps = co_attacker.total_steps
    print(f"\n[HOÀN TẤT BƯỚC 1] Split Learning + Co-adaptation xong sau {t_sl:.1f}s! Tổng số bước Co-adapted: {total_coadapt_steps} steps.")

    # Đánh giá cuối cùng của SL trên toàn bộ tập test
    final_test_loss, final_test_acc = evaluate_sl(
        client, server, test_full_loader, device, defense=defense, criterion=criterion_task
    )
    final_mse, final_coadapt_psnr, final_coadapt_ssim, final_coadapt_lpips = evaluate_coadapted_inversion(
        decoder=co_attacker.decoder,
        client=client,
        loader=test_full_loader,
        device=device,
        defense=defense,
        lpips_fn=lpips_fn,
    )

    # 6. NHÁNH B — FRESH ATTACKER VỚI CÙNG SỐ BƯỚC (MATCHED STEPS)
    print(f"\n[BƯỚC 2] Huấn luyện Nhánh B: Fresh Matched Steps (Khởi tạo từ đầu, chạy đúng {total_coadapt_steps} steps)...", flush=True)
    t_b_start = time.time()
    res_b = train_fresh_matched_steps(
        decoder_ctor=decoder_ctor,
        client=client,
        aux_loader=aux_loader,
        test_loader=test_full_loader,
        target_steps=total_coadapt_steps,
        device=device,
        defense=defense,
        lr=args.decoder_lr,
        eval_freq_steps=len(aux_loader) * args.eval_freq,
        lpips_fn=lpips_fn,
    )
    t_b = time.time() - t_b_start
    print(f"[HOÀN TẤT NHÁNH B] Xong sau {t_b:.1f}s | Final PSNR: {res_b['final_psnr']:.2f} dB | SSIM: {res_b['final_ssim']:.4f}")

    # 7. NHÁNH C — FRESH ATTACKER 30 EPOCHS CHUẨN
    print(f"\n[BƯỚC 3] Huấn luyện Nhánh C: Fresh 30 Epochs (Giao thức chuẩn lịch sử)...", flush=True)
    t_c_start = time.time()
    res_c = train_fresh_epochs(
        decoder_ctor=decoder_ctor,
        client=client,
        aux_loader=aux_loader,
        test_loader=test_full_loader,
        epochs=args.fresh_30ep,
        device=device,
        defense=defense,
        lr=args.decoder_lr,
        eval_freq=5,
        lpips_fn=lpips_fn,
    )
    t_c = time.time() - t_c_start
    print(f"[HOÀN TẤT NHÁNH C] Xong sau {t_c:.1f}s | Final PSNR: {res_c['final_psnr']:.2f} dB | SSIM: {res_c['final_ssim']:.4f}")

    # 8. Tính toán khoảng cách (Effect size of Co-adaptation)
    delta_ssim_final = final_coadapt_ssim - res_b["final_ssim"]
    delta_ssim_peak = best_coadapt_ssim - res_b["best_ssim"]
    delta_psnr_final = final_coadapt_psnr - res_b["final_psnr"]

    is_significant = delta_ssim_final > 0.05
    verdict = "Co-adapted Đột phá (SSIM tăng > 0.05)" if is_significant else "Bất biến với Co-adaptation (Chênh lệch không đáng kể)"

    # 9. Xuất ảnh lưới đối chứng và đồ thị
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    grid_path = os.path.join(out_dir, f"grid_coadapted_vs_fresh_{def_name}_s{seed}.png")
    save_3branch_comparison_grid(
        client=client,
        defense=defense,
        dec_coadapted=co_attacker.decoder,
        dec_matched=res_b["decoder"],
        dec_30ep=res_c["decoder"],
        loader=test_full_loader,
        device=device,
        save_path=grid_path,
        num_images=6,
    )

    curve_path = os.path.join(out_dir, f"curve_coadapted_{def_name}_s{seed}.png")
    plot_coadapted_curves(history_coadapt, curve_path)

    # 10. Tổng hợp record kết quả
    record = {
        "scenario_id": scenario_id,
        "defense": def_name.upper(),
        "seed": seed,
        "sl_epochs": args.epochs,
        "task_test_acc": round(final_test_acc * 100, 2),
        "total_steps": total_coadapt_steps,
        # Nhánh A
        "coadapt_final_psnr": round(final_coadapt_psnr, 2),
        "coadapt_final_ssim": round(final_coadapt_ssim, 4),
        "coadapt_final_lpips": round(final_coadapt_lpips, 4) if final_coadapt_lpips is not None else None,
        "coadapt_peak_psnr": round(best_coadapt_psnr, 2),
        "coadapt_peak_ssim": round(best_coadapt_ssim, 4),
        "coadapt_peak_epoch": best_coadapt_ep,
        # Nhánh B (Matched Steps)
        "fresh_matched_psnr": round(res_b["final_psnr"], 2),
        "fresh_matched_ssim": round(res_b["final_ssim"], 4),
        "fresh_matched_lpips": round(res_b["final_lpips"], 4) if res_b["final_lpips"] is not None else None,
        # Nhánh C (30 Epochs)
        "fresh_30ep_psnr": round(res_c["final_psnr"], 2),
        "fresh_30ep_ssim": round(res_c["final_ssim"], 4),
        "fresh_30ep_lpips": round(res_c["final_lpips"], 4) if res_c["final_lpips"] is not None else None,
        # So sánh khoa học
        "delta_ssim_vs_matched": round(delta_ssim_final, 4),
        "delta_ssim_peak_vs_matched": round(delta_ssim_peak, 4),
        "delta_psnr_vs_matched": round(delta_psnr_final, 2),
        "verdict": verdict,
        "grid_image": os.path.basename(grid_path),
        "history_coadapt": history_coadapt,
    }

    print("\n" + "-" * 90)
    print(f"KẾT QUẢ ĐỐI SÁNH 3 NHÁNH ({scenario_id.upper()}):")
    print(f"  * Task Test Acc (SL):     {record['task_test_acc']}%")
    print(f"  * Nhánh A (Co-adapted):   PSNR={record['coadapt_final_psnr']} dB | SSIM={record['coadapt_final_ssim']} (Peak: {record['coadapt_peak_psnr']}dB / {record['coadapt_peak_ssim']})")
    print(f"  * Nhánh B (Fresh-Matched):PSNR={record['fresh_matched_psnr']} dB | SSIM={record['fresh_matched_ssim']}")
    print(f"  * Nhánh C (Fresh-30ep):   PSNR={record['fresh_30ep_psnr']} dB | SSIM={record['fresh_30ep_ssim']}")
    print(f"  => HIỆU ỨNG CO-ADAPTATION: Delta SSIM = {record['delta_ssim_vs_matched']:+.4f} | Delta PSNR = {record['delta_psnr_vs_matched']:+.2f} dB")
    print(f"  => NHẬN ĐỊNH:             {verdict}")
    print("-" * 90 + "\n", flush=True)

    return record


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    defenses = [d.strip().lower() for d in args.defenses.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("=" * 90)
    print("THỰC NGHIỆM ĐỐI SÁNH: CO-ADAPTED THỤ ĐỘNG vs FRESH ATTACKER (3 NHÁNH ĐỐI CHỨNG)")
    print(f"Thiết bị chạy: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Danh sách Defenses: {defenses} | Seeds: {seeds}")
    print(f"Thư mục kết quả: {out_dir}")
    print("=" * 90, flush=True)

    lpips_fn = get_lpips_fn(device=device)

    results_file = os.path.join(out_dir, "results_coadapted_passive.json")
    all_records = []
    if args.resume and os.path.isfile(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                all_records = json.load(f)
        except Exception:
            all_records = []

    completed_ids = {r.get("scenario_id") for r in all_records if "scenario_id" in r}

    for d in defenses:
        for s in seeds:
            s_id = f"{d}_seed{s}"
            if args.resume and s_id in completed_ids:
                print(f"[RESUME] Bỏ qua {s_id} vì đã có kết quả!")
                continue

            rec = run_single_scenario(d, s, args, device, lpips_fn)
            all_records = [r for r in all_records if r.get("scenario_id") != s_id] + [rec]

            # Lưu ngay lập tức sau mỗi kịch bản
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2)

            # Xuất CSV tổng kết
            csv_file = os.path.join(out_dir, "results_coadapted_passive.csv")
            csv_fields = [
                "scenario_id", "defense", "seed", "task_test_acc", "total_steps",
                "coadapt_final_psnr", "coadapt_final_ssim", "coadapt_peak_psnr", "coadapt_peak_ssim",
                "fresh_matched_psnr", "fresh_matched_ssim",
                "fresh_30ep_psnr", "fresh_30ep_ssim",
                "delta_ssim_vs_matched", "delta_psnr_vs_matched", "verdict"
            ]
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_records)

            if args.backup_dir and os.path.isdir(args.backup_dir):
                import shutil
                shutil.copy2(results_file, os.path.join(args.backup_dir, os.path.basename(results_file)))
                shutil.copy2(csv_file, os.path.join(args.backup_dir, os.path.basename(csv_file)))
                print(f"  [DRIVE SYNC] Đã sao lưu sang: {args.backup_dir}", flush=True)

    # In bảng tổng hợp
    print("\n" + "=" * 115)
    print("BẢNG TỔNG HỢP: HIỆU ỨNG CO-ADAPTATION TRÊN CÁC BASELINE")
    print("=" * 115)
    header = (
        f"{'Scenario':<12} | {'Task Acc':>8} | {'Co-Adapt Final':>16} | "
        f"{'Co-Adapt Peak':>16} | {'Fresh Matched':>16} | {'Fresh 30ep':>14} | {'Δ SSIM':>8} | {'Kết luận'}"
    )
    print(header)
    print("-" * 115)
    for r in all_records:
        co_f = f"{r['coadapt_final_psnr']}dB / {r['coadapt_final_ssim']:.4f}"
        co_p = f"{r['coadapt_peak_psnr']}dB / {r['coadapt_peak_ssim']:.4f}"
        fr_m = f"{r['fresh_matched_psnr']}dB / {r['fresh_matched_ssim']:.4f}"
        fr_3 = f"{r['fresh_30ep_psnr']}dB / {r['fresh_30ep_ssim']:.4f}"
        print(
            f"{r['scenario_id']:<12} | {r['task_test_acc']:7.2f}% | {co_f:>16} | "
            f"{co_p:>16} | {fr_m:>16} | {fr_3:>14} | {r['delta_ssim_vs_matched']:+8.4f} | {r['verdict']}"
        )
    print("=" * 115 + "\n", flush=True)


if __name__ == "__main__":
    main()
