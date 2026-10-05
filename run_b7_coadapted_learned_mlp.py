#!/usr/bin/env python3
# ==============================================================================
# ĐỐI CHUẨN CO-ADAPTED THỤ ĐỘNG vs FRESH TRÊN B7 (LIGHTSPLIT) VỚI LEARNED MLP
# ==============================================================================
# - Baseline: B7 (Fixed Orthogonal Projection, Mode F, k=1024)
# - Attacker: Learned Adaptive MLP Unprojector (k -> 512 -> 65536 + ConvDecoder)
# - Nhánh A: Co-adapted Online (huấn luyện song song qua 100 epochs SL)
# - Nhánh B: Fresh Matched Steps (huấn luyện trên client B7 đã đóng băng, đúng 2000 steps)
# - Số seeds: 3 seeds (42, 7, 2024)
# - Điểm chốt: Lấy trung bình 10 epochs cuối (Epoch 91 -> 100) để tính Delta công bằng nhất!
# ==============================================================================
import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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

from src.models import ClientModel, ServerModel
from src.data.cifar import CIFAR10_MEAN, CIFAR10_STD
from src.defenses import FixedOrthoProjection
from src.attacks import Decoder
from src.attacks.coadapted import get_coadapted_data_splits
from src.metrics.reconstruction import psnr_ssim, get_lpips_fn, calculate_lpips, denormalize
from src.training.split_learning import evaluate_sl


class LearnedAdaptiveDecoder(nn.Module):
    """
    Attacker Learned Adaptive MLP Unprojector cho Baseline B7:
    zt in R^{B x k} -> unprojector (k -> hidden_dim -> D) -> reshape (B, 64, 32, 32) -> ConvDecoder -> x_rec
    """
    def __init__(self, k=1024, D=65536, hidden_dim=512):
        super().__init__()
        self.k = k
        self.D = D
        self.unprojector = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D),
            nn.BatchNorm1d(D)
        )
        self.conv_decoder = Decoder(in_channels=64, out_channels=3)

    def forward(self, zt):
        if zt.dim() > 2:
            zt = zt.view(zt.size(0), -1)
        z_unproj = self.unprojector(zt).view(-1, 64, 32, 32)
        return self.conv_decoder(z_unproj)


@torch.no_grad()
def evaluate_inversion(decoder, client, proj, loader, device, lpips_fn=None):
    """
    Đánh giá chất lượng tái tạo ảnh của Decoder trên tập loader kiểm thử.
    """
    client.eval()
    proj.eval()
    decoder.eval()

    total_samples = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    has_lpips = lpips_fn is not None

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        b = x.size(0)

        z = client(x)
        zt = proj(z)
        x_rec = decoder(zt)

        p, s = psnr_ssim(x, x_rec, CIFAR10_MEAN, CIFAR10_STD)
        psnr_sum += p * b
        ssim_sum += s * b

        if has_lpips:
            lp = calculate_lpips(x, x_rec, CIFAR10_MEAN, CIFAR10_STD, lpips_fn)
            if lp is not None:
                lpips_sum += lp * b
        total_samples += b

    avg_psnr = psnr_sum / max(total_samples, 1)
    avg_ssim = ssim_sum / max(total_samples, 1)
    avg_lpips = (lpips_sum / max(total_samples, 1)) if has_lpips else None
    return avg_psnr, avg_ssim, avg_lpips


@torch.no_grad()
def save_b7_comparison_grid(client, proj, dec_a, dec_b, loader, device, save_path, num_images=6):
    """
    Xuất lưới ảnh đối chứng: Ảnh gốc, Nhánh A (Co-adapted), Nhánh B (Fresh-Matched)
    """
    client.eval()
    proj.eval()
    dec_a.eval()
    dec_b.eval()

    imgs = []
    for x, _ in loader:
        imgs.append(x)
        if sum(t.size(0) for t in imgs) >= num_images:
            break

    x_all = torch.cat(imgs, dim=0)[:num_images].to(device)
    z = client(x_all)
    zt = proj(z)

    rec_a = dec_a(zt)
    rec_b = dec_b(zt)

    mean, std = CIFAR10_MEAN, CIFAR10_STD
    x_orig_np = denormalize(x_all, mean, std).cpu().numpy()
    x_a_np = denormalize(rec_a, mean, std).cpu().numpy()
    x_b_np = denormalize(rec_b, mean, std).cpu().numpy()

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2.4, 7.5))
    row_labels = [
        "Ảnh Gốc\nx",
        "Nhánh A\n(Co-adapted)",
        "Nhánh B\n(Fresh-Matched)"
    ]

    for i in range(num_images):
        orig = np.clip(x_orig_np[i].transpose(1, 2, 0), 0.0, 1.0)
        img_a = np.clip(x_a_np[i].transpose(1, 2, 0), 0.0, 1.0)
        img_b = np.clip(x_b_np[i].transpose(1, 2, 0), 0.0, 1.0)

        # Hàng 1
        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Mẫu #{i+1}", fontsize=11, fontweight="bold")

        # Hàng 2: Co-adapted
        p_a = peak_signal_noise_ratio(orig, img_a, data_range=1.0)
        s_a = structural_similarity(orig, img_a, channel_axis=2, data_range=1.0)
        axes[1, i].imshow(img_a)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"{p_a:.1f}dB | {s_a:.2f}", fontsize=10, color="#d62728" if s_a > 0.4 else "#2ca02c")

        # Hàng 3: Fresh Matched
        p_b = peak_signal_noise_ratio(orig, img_b, data_range=1.0)
        s_b = structural_similarity(orig, img_b, channel_axis=2, data_range=1.0)
        axes[2, i].imshow(img_b)
        axes[2, i].axis("off")
        axes[2, i].set_title(f"{p_b:.1f}dB | {s_b:.2f}", fontsize=10, color="#d62728" if s_b > 0.4 else "#2ca02c")

    for r in range(3):
        axes[r, 0].text(-0.25, 0.5, row_labels[r], transform=axes[r, 0].transAxes,
                        fontsize=11, fontweight="bold", va="center", ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[GRID] Đã xuất lưới ảnh đối chứng tại: {save_path}", flush=True)


def plot_b7_curves(history_a, history_b, save_path):
    """
    Vẽ đồ thị tiến trình hội tụ của Nhánh A và Nhánh B:
    Có tô vùng xám mờ highlight 10 epochs cuối (Epoch 91 -> 100)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. PSNR Curve
    ep_a = [h["epoch"] for h in history_a]
    psnr_a = [h["psnr"] for h in history_a]
    axes[0].plot(ep_a, psnr_a, marker="o", color="#1f77b4", label="Nhánh A (Co-adapted)", linewidth=2.0)

    if history_b:
        ep_b = [h["equiv_epoch"] for h in history_b]
        psnr_b = [h["psnr"] for h in history_b]
        axes[0].plot(ep_b, psnr_b, marker="s", color="#ff7f0e", linestyle="--", label="Nhánh B (Fresh-Matched)", linewidth=2.0)

    axes[0].axvspan(91, 100, color="gray", alpha=0.2, label="10 Epochs Cuối (Vùng tính Δ)")
    axes[0].set_title("So sánh PSNR (B7 Learned MLP)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch tương đương")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. SSIM Curve
    ssim_a = [h["ssim"] for h in history_a]
    axes[1].plot(ep_a, ssim_a, marker="o", color="#2ca02c", label="Nhánh A (Co-adapted)", linewidth=2.0)

    if history_b:
        ssim_b = [h["ssim"] for h in history_b]
        axes[1].plot(ep_b, ssim_b, marker="s", color="#d62728", linestyle="--", label="Nhánh B (Fresh-Matched)", linewidth=2.0)

    axes[1].axvspan(91, 100, color="gray", alpha=0.2, label="10 Epochs Cuối (Vùng tính Δ)")
    axes[1].axhline(0.40, color="#b2182b", linestyle=":", label="Ngưỡng SSIM 0.40")
    axes[1].set_title("So sánh SSIM (B7 Learned MLP)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch tương đương")
    axes[1].set_ylabel("SSIM")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Đã xuất đồ thị tại: {save_path}", flush=True)


def run_single_b7_seed(seed, args, device, lpips_fn):
    scenario_id = f"b7_learned_mlp_s{seed}"
    print("\n" + "=" * 90)
    print(f"BẮT ĐẦU KỊCH BẢN: {scenario_id.upper()} | SEED={seed}")
    print(f"Cơ chế: B7 (LightSplit Mode F, k={args.k}) | Attacker: Learned Adaptive MLP ({args.k}->{args.hidden_dim}->65536)")
    print(f"Epochs SL: {args.epochs} | D_aux: {args.aux_size} mẫu | Test Eval: {args.test_eval_size} mẫu")
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

    # 2. Khởi tạo Client, Server, B7 Projection
    client = ClientModel().to(device)
    server = ServerModel().to(device)
    proj = FixedOrthoProjection(D=64 * 32 * 32, k=args.k, seed=seed, device=device).to(device)

    # Optimizers cho Split Learning
    opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched_c = CosineAnnealingLR(opt_c, T_max=args.epochs)
    sched_s = CosineAnnealingLR(opt_s, T_max=args.epochs)
    criterion_task = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    # 3. Khởi tạo Attacker Nhánh A (Learned Adaptive MLP)
    dec_a = LearnedAdaptiveDecoder(k=args.k, D=64 * 32 * 32, hidden_dim=args.hidden_dim).to(device)
    opt_dec_a = torch.optim.Adam(dec_a.parameters(), lr=args.decoder_lr, weight_decay=1e-5)

    print(f"\n[BƯỚC 1] Huấn luyện Split Learning 100 Epochs đồng bộ Nhánh A (Co-adapted Online)...", flush=True)
    history_a = []
    last10_a_psnr = []
    last10_a_ssim = []
    best_a_psnr, best_a_ssim, best_a_ep = 0.0, 0.0, 0
    steps_per_epoch = len(aux_loader)

    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        # 1.1 SL 1 epoch
        client.train()
        server.train()
        proj.train()

        sl_loss_sum, sl_correct, sl_samples = 0.0, 0, 0
        for x, y in sl_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            b_size = x.size(0)

            opt_s.zero_grad()
            opt_c.zero_grad()

            z = client(x)
            zt = proj(z)
            z_d = zt.detach().requires_grad_(True)

            # Server Mode F: lift zt -> z_hat (không tham số)
            z_hat = proj.lift(z_d).view(b_size, 64, 32, 32)
            logits = server(z_hat)
            loss_task = criterion_task(logits, y)
            loss_task.backward()
            opt_s.step()

            # Gradient gửi về client qua kênh chiếu trực giao
            grad_zt = z_d.grad
            zt.backward(grad_zt)
            opt_c.step()

            sl_loss_sum += loss_task.item() * b_size
            sl_correct += (logits.argmax(1) == y).sum().item()
            sl_samples += b_size

        sched_c.step()
        sched_s.step()

        # 1.2 Nhánh A bước 1 epoch trên D_aux
        client.eval()
        proj.eval()
        dec_a.train()

        for x_aux, _ in aux_loader:
            x_aux = x_aux.to(device, non_blocking=True)
            with torch.no_grad():
                z_aux = client(x_aux)
                zt_aux = proj(z_aux)

            zt_input = zt_aux.detach()
            opt_dec_a.zero_grad()
            x_rec = dec_a(zt_input)
            loss_rec = criterion_mse(x_rec, x_aux)
            loss_rec.backward()
            opt_dec_a.step()

        # 1.3 Đánh giá định kỳ hoặc ở 10 epochs cuối
        is_last10 = (ep > (args.epochs - 10))
        should_eval = (ep == 1 or ep % args.eval_freq == 0 or ep == args.epochs or is_last10)

        if should_eval:
            # Task Test Acc (chỉ tính định kỳ để tối ưu tốc độ)
            task_acc = None
            if ep % args.eval_freq == 0 or ep == args.epochs:
                # Đánh giá SL qua projection
                client.eval()
                proj.eval()
                server.eval()
                corr, total = 0, 0
                with torch.no_grad():
                    for xt, yt in test_eval_loader:
                        xt, yt = xt.to(device), yt.to(device)
                        zt_eval = proj(client(xt))
                        zhat_eval = proj.lift(zt_eval).view(xt.size(0), 64, 32, 32)
                        preds = server(zhat_eval).argmax(1)
                        corr += (preds == yt).sum().item()
                        total += xt.size(0)
                task_acc = corr / total

            psnr_val, ssim_val, _ = evaluate_inversion(dec_a, client, proj, test_eval_loader, device)

            if psnr_val > best_a_psnr:
                best_a_psnr = psnr_val
                best_a_ssim = ssim_val
                best_a_ep = ep

            record = {
                "epoch": ep,
                "task_acc": task_acc,
                "psnr": round(psnr_val, 2),
                "ssim": round(ssim_val, 4),
            }
            history_a.append(record)

            if is_last10:
                last10_a_psnr.append(psnr_val)
                last10_a_ssim.append(ssim_val)

            tag_last10 = " [LAST 10]" if is_last10 else ""
            acc_str = f"Task Acc: {task_acc*100:.2f}% | " if task_acc is not None else ""
            print(f"  [Ep {ep:3d}/{args.epochs}] {acc_str}Nhánh A PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}{tag_last10} (Peak: {best_a_psnr:.2f}dB Ep {best_a_ep})", flush=True)

    t_a = time.time() - t_start
    total_steps_a = args.epochs * steps_per_epoch
    print(f"\n[HOÀN TẤT BƯỚC 1] Split Learning + Co-adaptation xong sau {t_a:.1f}s! Tổng steps: {total_steps_a}.", flush=True)

    # 4. Đánh giá Task Acc cuối cùng trên Full Test Set
    client.eval()
    proj.eval()
    server.eval()
    corr_full, total_full = 0, 0
    with torch.no_grad():
        for xt, yt in test_full_loader:
            xt, yt = xt.to(device), yt.to(device)
            zt_eval = proj(client(xt))
            zhat_eval = proj.lift(zt_eval).view(xt.size(0), 64, 32, 32)
            preds = server(zhat_eval).argmax(1)
            corr_full += (preds == yt).sum().item()
            total_full += xt.size(0)
    final_task_acc = corr_full / total_full

    # Đóng băng tuyệt đối Client và Proj
    client.eval()
    proj.eval()
    for p in client.parameters():
        p.requires_grad = False
    for p in proj.parameters():
        p.requires_grad = False

    # 5. Huấn luyện Nhánh B: Fresh Matched Steps (Đúng total_steps_a = 2000 steps)
    print(f"\n[BƯỚC 2] Huấn luyện Nhánh B: Fresh Matched Steps (Khởi tạo từ đầu, chạy đúng {total_steps_a} steps)...", flush=True)
    dec_b = LearnedAdaptiveDecoder(k=args.k, D=64 * 32 * 32, hidden_dim=args.hidden_dim).to(device)
    opt_dec_b = torch.optim.Adam(dec_b.parameters(), lr=args.decoder_lr, weight_decay=1e-5)

    history_b = []
    last10_b_psnr = []
    last10_b_ssim = []
    best_b_psnr, best_b_ssim, best_b_step = 0.0, 0.0, 0

    t_start_b = time.time()
    step_count = 0
    cur_epoch_b = 0

    while step_count < total_steps_a:
        for x_aux, _ in aux_loader:
            if step_count >= total_steps_a:
                break

            x_aux = x_aux.to(device, non_blocking=True)
            with torch.no_grad():
                z_aux = client(x_aux)
                zt_aux = proj(z_aux)

            zt_input = zt_aux.detach()
            dec_b.train()
            opt_dec_b.zero_grad()
            x_rec = dec_b(zt_input)
            loss_rec = criterion_mse(x_rec, x_aux)
            loss_rec.backward()
            opt_dec_b.step()

            step_count += 1

            # Đánh giá theo mốc epoch tương đương (mỗi steps_per_epoch = 1 epoch)
            if step_count % steps_per_epoch == 0:
                cur_epoch_b += 1
                is_last10_b = (cur_epoch_b > (args.epochs - 10))
                should_eval_b = (cur_epoch_b == 1 or cur_epoch_b % args.eval_freq == 0 or cur_epoch_b == args.epochs or is_last10_b)

                if should_eval_b:
                    psnr_b, ssim_b, _ = evaluate_inversion(dec_b, client, proj, test_eval_loader, device)

                    if psnr_b > best_b_psnr:
                        best_b_psnr = psnr_b
                        best_b_ssim = ssim_b
                        best_b_step = step_count

                    history_b.append({
                        "step": step_count,
                        "equiv_epoch": cur_epoch_b,
                        "psnr": round(psnr_b, 2),
                        "ssim": round(ssim_b, 4),
                    })

                    if is_last10_b:
                        last10_b_psnr.append(psnr_b)
                        last10_b_ssim.append(ssim_b)

                    tag_last10 = " [LAST 10]" if is_last10_b else ""
                    print(f"  [Step {step_count:4d}/{total_steps_a} (Ep {cur_epoch_b:3d})] Nhánh B PSNR: {psnr_b:.2f} dB | SSIM: {ssim_b:.4f}{tag_last10} (Peak: {best_b_psnr:.2f}dB Step {best_b_step})", flush=True)

    t_b = time.time() - t_start_b
    print(f"[HOÀN TẤT NHÁNH B] Xong sau {t_b:.1f}s!", flush=True)

    # 6. Đánh giá kiểm định cuối cùng với LPIPS trên toàn bộ Test Set
    final_psnr_a, final_ssim_a, final_lpips_a = evaluate_inversion(dec_a, client, proj, test_full_loader, device, lpips_fn=lpips_fn)
    final_psnr_b, final_ssim_b, final_lpips_b = evaluate_inversion(dec_b, client, proj, test_full_loader, device, lpips_fn=lpips_fn)

    # 7. Tính toán Trung bình 10 Epochs Cuối (Last-10 Average)
    mean_a_psnr_10 = float(np.mean(last10_a_psnr)) if last10_a_psnr else final_psnr_a
    std_a_psnr_10 = float(np.std(last10_a_psnr)) if last10_a_psnr else 0.0
    mean_a_ssim_10 = float(np.mean(last10_a_ssim)) if last10_a_ssim else final_ssim_a
    std_a_ssim_10 = float(np.std(last10_a_ssim)) if last10_a_ssim else 0.0

    mean_b_psnr_10 = float(np.mean(last10_b_psnr)) if last10_b_psnr else final_psnr_b
    std_b_psnr_10 = float(np.std(last10_b_psnr)) if last10_b_psnr else 0.0
    mean_b_ssim_10 = float(np.mean(last10_b_ssim)) if last10_b_ssim else final_ssim_b
    std_b_ssim_10 = float(np.std(last10_b_ssim)) if last10_b_ssim else 0.0

    delta_ssim_last10 = mean_a_ssim_10 - mean_b_ssim_10
    delta_psnr_last10 = mean_a_psnr_10 - mean_b_psnr_10
    delta_ssim_final = final_ssim_a - final_ssim_b
    delta_psnr_final = final_psnr_a - final_psnr_b

    # Kết luận định tính theo ngưỡng quyết định 0.05
    if delta_ssim_last10 > 0.05:
        verdict = "Co-adapted có ưu thế thực sự (Delta SSIM > 0.05)"
    elif delta_ssim_last10 < -0.05:
        verdict = "Fresh Attacker vượt trội hơn (Co-adaptation bị tụt hậu)"
    else:
        verdict = "Bất biến với Co-adaptation (Chênh lệch không đáng kể <= 0.05)"

    print("\n" + "-" * 90)
    print(f"KẾT QUẢ ĐỐI SÁNH TRUNG BÌNH 10 EPOCHS CUỐI - SEED {seed}:")
    print(f"  * Task Test Acc (SL):             {final_task_acc*100:.2f}%")
    print(f"  * Nhánh A Last-10 (Co-adapted):   PSNR = {mean_a_psnr_10:.2f} ± {std_a_psnr_10:.2f} dB | SSIM = {mean_a_ssim_10:.4f} ± {std_a_ssim_10:.4f}")
    print(f"  * Nhánh B Last-10 (Fresh-Match):  PSNR = {mean_b_psnr_10:.2f} ± {std_b_psnr_10:.2f} dB | SSIM = {mean_b_ssim_10:.4f} ± {std_b_ssim_10:.4f}")
    print(f"  * Đỉnh cao nhất (Peak):           Nhánh A Peak = {best_a_psnr:.2f} dB / {best_a_ssim:.4f} | Nhánh B Peak = {best_b_psnr:.2f} dB / {best_b_ssim:.4f}")
    print(f"  * Mốc Final (Ep 100 vs Step 2000):Nhánh A = {final_psnr_a:.2f} dB / {final_ssim_a:.4f} | Nhánh B = {final_psnr_b:.2f} dB / {final_ssim_b:.4f}")
    print(f"  ==> CHÊNH LỆCH TRUNG BÌNH 10 EPOCHS CUỐI:")
    print(f"      Delta SSIM (Last-10) = {delta_ssim_last10:+.4f} | Delta PSNR (Last-10) = {delta_psnr_last10:+.2f} dB")
    print(f"      Delta SSIM (Final)   = {delta_ssim_final:+.4f} | Delta PSNR (Final)   = {delta_psnr_final:+.2f} dB")
    print(f"  ==> KẾT LUẬN: {verdict}")
    print("-" * 90, flush=True)

    # 8. Xuất ảnh và đồ thị
    grid_path = os.path.join(args.output_dir, f"grid_b7_learned_mlp_s{seed}.png")
    save_b7_comparison_grid(client, proj, dec_a, dec_b, test_full_loader, device, grid_path, num_images=6)

    curve_path = os.path.join(args.output_dir, f"curve_b7_learned_mlp_s{seed}.png")
    plot_b7_curves(history_a, history_b, curve_path)

    scenario_res = {
        "scenario_id": scenario_id,
        "seed": seed,
        "k": args.k,
        "hidden_dim": args.hidden_dim,
        "task_test_acc": round(final_task_acc * 100, 2),
        "total_steps": total_steps_a,
        "mean_a_psnr_10": round(mean_a_psnr_10, 2),
        "std_a_psnr_10": round(std_a_psnr_10, 2),
        "mean_a_ssim_10": round(mean_a_ssim_10, 4),
        "std_a_ssim_10": round(std_a_ssim_10, 4),
        "mean_b_psnr_10": round(mean_b_psnr_10, 2),
        "std_b_psnr_10": round(std_b_psnr_10, 2),
        "mean_b_ssim_10": round(mean_b_ssim_10, 4),
        "std_b_ssim_10": round(std_b_ssim_10, 4),
        "delta_ssim_last10": round(delta_ssim_last10, 4),
        "delta_psnr_last10": round(delta_psnr_last10, 2),
        "final_a_psnr": round(final_psnr_a, 2),
        "final_a_ssim": round(final_ssim_a, 4),
        "final_b_psnr": round(final_psnr_b, 2),
        "final_b_ssim": round(final_ssim_b, 4),
        "delta_ssim_final": round(delta_ssim_final, 4),
        "delta_psnr_final": round(delta_psnr_final, 2),
        "peak_a_psnr": round(best_a_psnr, 2),
        "peak_a_ssim": round(best_a_ssim, 4),
        "peak_b_psnr": round(best_b_psnr, 2),
        "peak_b_ssim": round(best_b_ssim, 4),
        "lpips_a": round(final_lpips_a, 4) if final_lpips_a else None,
        "lpips_b": round(final_lpips_b, 4) if final_lpips_b else None,
        "verdict": verdict,
        "history_a": history_a,
        "history_b": history_b,
        "last10_a_ssim": [round(s, 4) for s in last10_a_ssim],
        "last10_b_ssim": [round(s, 4) for s in last10_b_ssim],
    }

    return scenario_res


def main():
    parser = argparse.ArgumentParser(description="Đối chuẩn Co-adapted vs Fresh trên B7 với Learned Adaptive MLP")
    parser.add_argument("--k", type=int, default=1024, help="Chiều chiếu trực giao k của B7 (mặc định: 1024)")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Chiều ẩn MLP unprojector (mặc định: 512)")
    parser.add_argument("--seeds", type=str, default="42,7,2024", help="Danh sách seeds (mặc định: '42,7,2024')")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs Split Learning (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate SGD cho SL (mặc định: 0.1)")
    parser.add_argument("--decoder-lr", type=float, default=1e-3, help="Learning rate Adam cho Decoder (mặc định: 1e-3)")
    parser.add_argument("--aux-size", type=int, default=2500, help="Kích thước tập ảnh phụ cố định D_aux (mặc định: 2500)")
    parser.add_argument("--test-eval-size", type=int, default=2000, help="Kích thước tập test nhanh (mặc định: 2000)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá định kỳ (mặc định: 5)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_CoAdapted_B7_Learned"))
    parser.add_argument("--backup-dir", type=str, default=None, help="Thư mục sao lưu Google Drive nếu chạy Colab")
    parser.add_argument("--resume", action="store_true", default=False, help="Bỏ qua các seed đã hoàn thành")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Chạy thử 2 epochs để kiểm thử mã")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 2
        args.eval_freq = 1
        args.seeds = "42"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    lpips_fn = get_lpips_fn(device=device)

    print("=" * 90)
    print("THỰC NGHIỆM ĐỐI CHUẨN KHOA HỌC: B7 (LIGHTSPLIT) VỚI LEARNED ADAPTIVE MLP")
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Số seeds đánh giá: {seed_list} | Tổng số: {len(seed_list)} seeds")
    print(f"Đánh giá trọng tâm: ΔSSIM và ΔPSNR lấy TRUNG BÌNH 10 EPOCHS CUỐI (Epoch 91 -> 100)")
    print(f"Thư mục lưu trữ: {args.output_dir}")
    print("=" * 90, flush=True)

    json_path = os.path.join(args.output_dir, "results_b7_learned_mlp_last10.json")
    csv_path = os.path.join(args.output_dir, "results_b7_learned_mlp_last10.csv")

    all_results = {}
    if args.resume and os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"[RESUME] Đã nạp {len(all_results)} kết quả seed trước đó từ JSON.")
        except Exception:
            all_results = {}

    for seed in seed_list:
        scenario_id = f"b7_learned_mlp_s{seed}"
        if args.resume and scenario_id in all_results:
            print(f"[BỎ QUA] {scenario_id} đã có kết quả.")
            continue

        res = run_single_b7_seed(seed, args, device, lpips_fn)
        all_results[scenario_id] = res

        # Cập nhật JSON sau mỗi seed
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Cập nhật CSV sau mỗi seed
        with open(csv_path, "w", encoding="utf-8") as f:
            header = [
                "seed", "task_test_acc", "total_steps",
                "mean_a_psnr_10", "std_a_psnr_10", "mean_a_ssim_10", "std_a_ssim_10",
                "mean_b_psnr_10", "std_b_psnr_10", "mean_b_ssim_10", "std_b_ssim_10",
                "delta_ssim_last10", "delta_psnr_last10",
                "final_a_psnr", "final_a_ssim", "final_b_psnr", "final_b_ssim",
                "delta_ssim_final", "delta_psnr_final",
                "peak_a_psnr", "peak_a_ssim", "peak_b_psnr", "peak_b_ssim",
                "verdict"
            ]
            f.write(",".join(header) + "\n")
            for r in all_results.values():
                row = [
                    str(r["seed"]),
                    str(r["task_test_acc"]),
                    str(r["total_steps"]),
                    str(r["mean_a_psnr_10"]),
                    str(r["std_a_psnr_10"]),
                    str(r["mean_a_ssim_10"]),
                    str(r["std_a_ssim_10"]),
                    str(r["mean_b_psnr_10"]),
                    str(r["std_b_psnr_10"]),
                    str(r["mean_b_ssim_10"]),
                    str(r["std_b_ssim_10"]),
                    str(r["delta_ssim_last10"]),
                    str(r["delta_psnr_last10"]),
                    str(r["final_a_psnr"]),
                    str(r["final_a_ssim"]),
                    str(r["final_b_psnr"]),
                    str(r["final_b_ssim"]),
                    str(r["delta_ssim_final"]),
                    str(r["delta_psnr_final"]),
                    str(r["peak_a_psnr"]),
                    str(r["peak_a_ssim"]),
                    str(r["peak_b_psnr"]),
                    str(r["peak_b_ssim"]),
                    f'"{r["verdict"]}"'
                ]
                f.write(",".join(row) + "\n")

        # Đồng bộ Google Drive nếu có
        if args.backup_dir and os.path.exists(args.backup_dir):
            import shutil
            shutil.copy2(json_path, os.path.join(args.backup_dir, os.path.basename(json_path)))
            shutil.copy2(csv_path, os.path.join(args.backup_dir, os.path.basename(csv_path)))
            grid_file = os.path.join(args.output_dir, f"grid_b7_learned_mlp_s{seed}.png")
            if os.path.exists(grid_file):
                shutil.copy2(grid_file, os.path.join(args.backup_dir, os.path.basename(grid_file)))
            curve_file = os.path.join(args.output_dir, f"curve_b7_learned_mlp_s{seed}.png")
            if os.path.exists(curve_file):
                shutil.copy2(curve_file, os.path.join(args.backup_dir, os.path.basename(curve_file)))
            print(f"[DRIVE SYNC] Đã đồng bộ kết quả Seed {seed} sang Google Drive: {args.backup_dir}")

    # ==============================================================================
    # BẢNG TỔNG HỢP VÀ KẾT LUẬN CUỐI CÙNG QUA 3 SEEDS
    # ==============================================================================
    print("\n" + "=" * 95)
    print("TỔNG HỢP KẾT QUẢ ĐỐI CHUẨN B7 VỚI LEARNED MLP QUA CÁC SEEDS:")
    print("=" * 95)
    print(f"{'Seed':<8} | {'Task Acc':<10} | {'Nhánh A (Last10)':<22} | {'Nhánh B (Last10)':<22} | {'Delta SSIM (Last10)':<20} | {'Delta PSNR':<10}")
    print("-" * 95)

    delta_ssims = []
    delta_psnrs = []
    task_accs = []
    a_ssims = []
    b_ssims = []

    for r in all_results.values():
        s = r["seed"]
        acc = r["task_test_acc"]
        a_str = f"{r['mean_a_psnr_10']:.2f}dB / {r['mean_a_ssim_10']:.4f}"
        b_str = f"{r['mean_b_psnr_10']:.2f}dB / {r['mean_b_ssim_10']:.4f}"
        d_ssim = r["delta_ssim_last10"]
        d_psnr = r["delta_psnr_last10"]

        delta_ssims.append(d_ssim)
        delta_psnrs.append(d_psnr)
        task_accs.append(acc)
        a_ssims.append(r["mean_a_ssim_10"])
        b_ssims.append(r["mean_b_ssim_10"])

        print(f"{s:<8} | {acc:.2f}%{'':<4} | {a_str:<22} | {b_str:<22} | {d_ssim:>+10.4f}{'':<10} | {d_psnr:>+7.2f} dB")

    print("-" * 95)
    if delta_ssims:
        mean_d_ssim = np.mean(delta_ssims)
        std_d_ssim = np.std(delta_ssims)
        mean_d_psnr = np.mean(delta_psnrs)
        std_d_psnr = np.std(delta_psnrs)
        mean_acc = np.mean(task_accs)
        std_acc = np.std(task_accs)

        print(f"{'TRUNG BÌNH':<8} | {mean_acc:.2f}±{std_acc:.2f}%{'':<2} | {'A: ' + str(round(np.mean(a_ssims), 4)):<22} | {'B: ' + str(round(np.mean(b_ssims), 4)):<22} | {mean_d_ssim:>+7.4f} ± {std_d_ssim:.4f}{'':<4} | {mean_d_psnr:>+5.2f} ± {std_d_psnr:.2f} dB")
        print("=" * 95)
        print(f"\n[KẾT LUẬN CHÍNH THỨC]:")
        if abs(mean_d_ssim) <= 0.05:
            print(f"  -> Ngay cả với Attacker Learned Adaptive MLP mạnh nhất trên Baseline B7 (dư địa lớn nhất),")
            print(f"     hiệu ứng Co-adaptation thụ động vẫn BẤT BIẾN: Delta SSIM = {mean_d_ssim:+.4f} <= 0.05 (ngưỡng nhiễu thống kê).")
            print(f"  -> Bác bỏ hoàn toàn giả thuyết rằng Co-adaptation thụ động đem lại lợi thế tái tạo so với Fresh Attacker.")
        else:
            print(f"  -> Co-adaptation thụ động tạo ra sự khác biệt đo lường được: Delta SSIM = {mean_d_ssim:+.4f}.")
    print("=" * 95 + "\n", flush=True)


if __name__ == "__main__":
    main()
