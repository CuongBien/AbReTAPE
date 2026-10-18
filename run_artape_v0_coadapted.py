#!/usr/bin/env python3
# ==============================================================================
# BƯỚC 3: ĐỐI CHUẨN KHOA HỌC: CO-ADAPTED THỤ ĐỘNG vs FRESH TRÊN AR-TAPE v0 (N_aux = 10.000)
# ==============================================================================
# - Mục tiêu: Xác định xem đối thủ Co-adapted (đồng thích ứng bám theo Client từng epoch)
#   có đạt được lợi thế tái tạo nào vượt trội so với đối thủ Fresh cùng ngân sách hay không.
# - Dữ liệu: N_aux = 10.000 ảnh phụ cố định; D_sl = 40.000 ảnh Split Learning
# - Nhánh A: Co-adapted Online (huấn luyện song song qua các epoch Split Learning trên D_aux)
# - Nhánh B: Fresh Matched Steps (huấn luyện trên client AR-TAPE đã đóng băng, đúng tổng số steps của Nhánh A)
# - Chỉ số cốt lõi: Δ = Trung bình 10 epochs cuối của Nhánh A trừ Nhánh B:
#   ΔSSIM = mean_last10(A) - mean_last10(B)
# ==============================================================================
import os
import sys
import time
import json
import csv
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
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

from src.data.cifar import CIFAR10_MEAN, CIFAR10_STD
from src.attacks.coadapted import get_coadapted_data_splits
from src.metrics import psnr_ssim, get_lpips_fn, calculate_lpips
from src.metrics.reconstruction import denormalize
from src.utils.visualize import save_reconstruction_grid

from train_artape_v0 import ClientWithBottleneck, ServerWithBottleneck, LearnedAdaptiveDecoder


@torch.no_grad()
def evaluate_inversion(decoder, client, loader, device, lpips_fn=None):
    client.eval()
    decoder.eval()

    total_samples = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    has_lpips = lpips_fn is not None

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        b = x.size(0)

        zt = client(x)
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
def save_comparison_grid(client, dec_a, dec_b, loader, device, save_path, num_images=8):
    client.eval()
    dec_a.eval()
    dec_b.eval()

    imgs = []
    for x, _ in loader:
        imgs.append(x)
        if sum(t.size(0) for t in imgs) >= num_images:
            break

    x_all = torch.cat(imgs, dim=0)[:num_images].to(device)
    zt = client(x_all)
    rec_a = dec_a(zt)
    rec_b = dec_b(zt)

    mean, std = CIFAR10_MEAN, CIFAR10_STD
    x_orig_np = denormalize(x_all, mean, std).cpu().numpy()
    x_a_np = denormalize(rec_a, mean, std).cpu().numpy()
    x_b_np = denormalize(rec_b, mean, std).cpu().numpy()

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2.2, 7.0))
    row_labels = ["Ảnh Gốc x", "Nhánh A\n(Co-adapted)", "Nhánh B\n(Fresh Matched)"]

    for i in range(num_images):
        axes[0, i].imshow(np.clip(x_orig_np[i].transpose(1, 2, 0), 0.0, 1.0))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title(row_labels[0], fontsize=10, loc="left")

        axes[1, i].imshow(np.clip(x_a_np[i].transpose(1, 2, 0), 0.0, 1.0))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title(row_labels[1], fontsize=10, loc="left")

        axes[2, i].imshow(np.clip(x_b_np[i].transpose(1, 2, 0), 0.0, 1.0))
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title(row_labels[2], fontsize=10, loc="left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_coadapted_single(args, device):
    mode_str = f"_{args.unprojector_mode}" if args.unprojector_mode != "mlp" else ""
    fix_str = "_fixR" if getattr(args, "fixed_bottleneck", False) else ""
    scenario_id = f"artape_v0_coadapted_k{args.k}{mode_str}{fix_str}_lam{args.lambda_val}_naux{args.aux_size}_s{args.seed}"
    print("=" * 90)
    print(f"KHỞI CHẠY THỰC NGHIỆM CO-ADAPTED TRÊN AR-TAPE v0 (N_aux = {args.aux_size})")
    print(f"Kịch bản: {scenario_id} | Seed: {args.seed} | Bottleneck k={args.k}")
    print(f"Unprojector Mode: {args.unprojector_mode} | Fixed Bottleneck: {getattr(args, 'fixed_bottleneck', False)}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Lambda: {args.lambda_val} (Warmup={getattr(args, 'warmup_epochs', 0)})")
    print("=" * 90, flush=True)

    # 1. Nạp dữ liệu splits Co-adapted: D_sl (40k), D_aux (10k), D_test_eval (2k), D_test_full (10k)
    sl_loader, aux_loader, test_eval_loader, test_full_loader = get_coadapted_data_splits(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        aux_size=args.aux_size,
        test_eval_size=args.test_eval_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # 2. Khởi tạo Client và Server AR-TAPE v0
    client = ClientWithBottleneck(
        k=args.k, D=65536, init_ortho=True, seed=args.seed, fixed_bottleneck=getattr(args, "fixed_bottleneck", False)
    ).to(device)
    ortho_q = client.R if client.fixed_bottleneck else client.bottleneck.weight.t().detach()
    server = ServerWithBottleneck(
        k=args.k, m=args.m, D=65536, unprojector_mode=args.unprojector_mode, ortho_matrix=ortho_q, seed=args.seed
    ).to(device)

    # Khởi tạo In-training Adversary cho Split Learning (min-max nội bộ)
    adv_sl = LearnedAdaptiveDecoder(k=args.k, D=65536, hidden_dim=args.m).to(device)

    # Khởi tạo Attacker Nhánh A: Co-adapted Shadow Decoder (học trên D_aux)
    dec_a = LearnedAdaptiveDecoder(k=args.k, D=65536, hidden_dim=args.m).to(device)

    # Optimizers
    opt_cs = torch.optim.SGD(list(client.parameters()) + list(server.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched_cs = MultiStepLR(opt_cs, milestones=[50, 75], gamma=0.1)

    opt_adv_sl = torch.optim.Adam(adv_sl.parameters(), lr=1e-3)
    opt_dec_a = torch.optim.Adam(dec_a.parameters(), lr=args.decoder_lr)

    criterion_task = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    history_a = []
    last10_a_psnr = []
    last10_a_ssim = []
    best_a_psnr, best_a_ssim, best_a_ep = 0.0, 0.0, 0
    best_a_state = copy.deepcopy(dec_a.state_dict())

    steps_per_epoch_aux = len(aux_loader)
    t_start = time.time()

    print("\n[BƯỚC 1/2] Huấn luyện Split Learning đồng bộ Nhánh A (Co-adapted Online)...", flush=True)

    for ep in range(1, args.epochs + 1):
        cur_lam = args.lambda_val * min(1.0, ep / max(1, args.warmup_epochs)) if getattr(args, "warmup_epochs", 0) > 0 else args.lambda_val

        # 1.1 Huấn luyện 1 epoch Split Learning trên D_sl (40k ảnh)
        client.train()
        server.train()
        adv_sl.train()

        for x, y in sl_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            zt = client(x)

            if cur_lam > 0:
                # Max step cho adversary nội bộ
                zt_det = zt.detach()
                x_rec_adv = adv_sl(zt_det)
                loss_adv = criterion_mse(x_rec_adv, x)
                opt_adv_sl.zero_grad()
                loss_adv.backward()
                opt_adv_sl.step()

                # Min step cho client & server
                for p in adv_sl.parameters():
                    p.requires_grad = False

                logits = server(zt)
                loss_task = criterion_task(logits, y)
                x_rec_c = adv_sl(zt)
                rec_mse = criterion_mse(x_rec_c, x)

                opt_cs.zero_grad()
                loss_task.backward(retain_graph=True)

                client_params = [p for p in client.parameters() if p.requires_grad]
                loss_priv = - cur_lam * torch.clamp(rec_mse, max=args.cap)
                grads_priv = torch.autograd.grad(loss_priv, client_params, retain_graph=False)

                for p, g in zip(client_params, grads_priv):
                    if p.grad is not None:
                        p.grad.add_(g)
                    else:
                        p.grad = g
                opt_cs.step()

                for p in adv_sl.parameters():
                    p.requires_grad = True
            else:
                logits = server(zt)
                loss_task = criterion_task(logits, y)
                opt_cs.zero_grad()
                loss_task.backward()
                opt_cs.step()

        sched_cs.step()

        # 1.2 Nhánh A bước 1 epoch trên D_aux (10.000 ảnh phụ)
        client.eval()
        dec_a.train()
        for x_aux, _ in aux_loader:
            x_aux = x_aux.to(device, non_blocking=True)
            with torch.no_grad():
                zt_aux = client(x_aux)

            opt_dec_a.zero_grad()
            x_rec = dec_a(zt_aux.detach())
            loss_rec = criterion_mse(x_rec, x_aux)
            loss_rec.backward()
            opt_dec_a.step()

        # 1.3 Đánh giá định kỳ & 10 epoch cuối
        is_last10 = (ep > (args.epochs - 10))
        should_eval = (ep == 1 or ep % args.eval_freq == 0 or ep == args.epochs or is_last10)

        if should_eval:
            psnr_val, ssim_val, _ = evaluate_inversion(dec_a, client, test_eval_loader, device)
            if ssim_val > best_a_ssim:
                best_a_ssim = ssim_val
                best_a_psnr = psnr_val
                best_a_ep = ep
                best_a_state = copy.deepcopy(dec_a.state_dict())

            record = {
                "epoch": ep,
                "psnr": round(psnr_val, 2),
                "ssim": round(ssim_val, 4)
            }
            history_a.append(record)

            if is_last10:
                last10_a_psnr.append(psnr_val)
                last10_a_ssim.append(ssim_val)

            tag_last10 = " [LAST 10]" if is_last10 else ""
            print(f"  [Ep {ep:3d}/{args.epochs}] Nhánh A PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}{tag_last10} (Peak: SSIM {best_a_ssim:.4f} / {best_a_psnr:.2f}dB Ep {best_a_ep})", flush=True)

    t_a = time.time() - t_start
    total_steps_a = args.epochs * steps_per_epoch_aux
    print(f"\n[BƯỚC 1 XONG] Nhánh A hoàn tất sau {t_a:.1f}s | Tổng steps trên D_aux: {total_steps_a}", flush=True)

    # Đóng băng Client tuyệt đối
    client.eval()
    for p in client.parameters():
        p.requires_grad = False

    # -----------------------------------------------------------------
    # BƯỚC 2: NHÁNH B - FRESH MATCHED STEPS (Huấn luyện từ đầu trên đúng total_steps_a)
    # -----------------------------------------------------------------
    print(f"\n[BƯỚC 2/2] Huấn luyện Nhánh B: Fresh Matched Steps (Đúng {total_steps_a} steps trên Client đã đóng băng)...", flush=True)
    dec_b = LearnedAdaptiveDecoder(k=args.k, D=65536, hidden_dim=args.m).to(device)
    opt_dec_b = torch.optim.Adam(dec_b.parameters(), lr=args.decoder_lr)

    history_b = []
    last10_b_psnr = []
    last10_b_ssim = []
    best_b_psnr, best_b_ssim, best_b_step = 0.0, 0.0, 0
    best_b_state = copy.deepcopy(dec_b.state_dict())

    t_start_b = time.time()
    step_count = 0
    cur_epoch_b = 0

    while step_count < total_steps_a:
        for x_aux, _ in aux_loader:
            if step_count >= total_steps_a:
                break

            x_aux = x_aux.to(device, non_blocking=True)
            with torch.no_grad():
                zt_aux = client(x_aux)

            dec_b.train()
            opt_dec_b.zero_grad()
            x_rec = dec_b(zt_aux.detach())
            loss_rec = criterion_mse(x_rec, x_aux)
            loss_rec.backward()
            opt_dec_b.step()

            step_count += 1

            # Đánh giá theo mốc tương đương 1 epoch (sau mỗi steps_per_epoch_aux)
            if step_count % steps_per_epoch_aux == 0:
                cur_epoch_b += 1
                is_last10_b = (cur_epoch_b > (args.epochs - 10))
                should_eval_b = (cur_epoch_b == 1 or cur_epoch_b % args.eval_freq == 0 or cur_epoch_b == args.epochs or is_last10_b)

                if should_eval_b:
                    psnr_b, ssim_b, _ = evaluate_inversion(dec_b, client, test_eval_loader, device)
                    if ssim_b > best_b_ssim:
                        best_b_ssim = ssim_b
                        best_b_psnr = psnr_b
                        best_b_step = step_count
                        best_b_state = copy.deepcopy(dec_b.state_dict())

                    record_b = {
                        "epoch_equiv": cur_epoch_b,
                        "step": step_count,
                        "psnr": round(psnr_b, 2),
                        "ssim": round(ssim_b, 4)
                    }
                    history_b.append(record_b)

                    if is_last10_b:
                        last10_b_psnr.append(psnr_b)
                        last10_b_ssim.append(ssim_b)

                    tag_last10_b = " [LAST 10]" if is_last10_b else ""
                    print(f"  [Step {step_count:4d}/{total_steps_a} ~ Ep {cur_epoch_b:3d}] Nhánh B PSNR: {psnr_b:.2f} dB | SSIM: {ssim_b:.4f}{tag_last10_b} (Peak: SSIM {best_b_ssim:.4f} / {best_b_psnr:.2f}dB)", flush=True)

    t_b = time.time() - t_start_b

    # 3. Lưu checkpoints cuối và tốt nhất
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(client.state_dict(), os.path.join(checkpoint_dir, f"{scenario_id}_client.pt"))
    torch.save(server.state_dict(), os.path.join(checkpoint_dir, f"{scenario_id}_server.pt"))
    torch.save(best_a_state, os.path.join(checkpoint_dir, f"{scenario_id}_dec_a_best.pt"))
    torch.save(best_b_state, os.path.join(checkpoint_dir, f"{scenario_id}_dec_b_best.pt"))
    torch.save(dec_a.state_dict(), os.path.join(checkpoint_dir, f"{scenario_id}_dec_a_final.pt"))
    torch.save(dec_b.state_dict(), os.path.join(checkpoint_dir, f"{scenario_id}_dec_b_final.pt"))

    # 4. Đánh giá chuyên sâu trên các checkpoint TỐT NHẤT (Peak validation SSIM)
    print("\n[ĐÁNH GIÁ CHUYÊN SÂU] Nạp checkpoints tốt nhất (Peak SSIM) để đánh giá không thiên vị...", flush=True)
    dec_a_best = LearnedAdaptiveDecoder(k=args.k, D=65536, hidden_dim=args.m).to(device)
    dec_a_best.load_state_dict(best_a_state)
    dec_b_best = LearnedAdaptiveDecoder(k=args.k, D=65536, hidden_dim=args.m).to(device)
    dec_b_best.load_state_dict(best_b_state)

    best_psnr_a_2k, best_ssim_a_2k, _ = evaluate_inversion(dec_a_best, client, test_eval_loader, device)
    best_psnr_b_2k, best_ssim_b_2k, _ = evaluate_inversion(dec_b_best, client, test_eval_loader, device)
    delta_ssim_best_2k = best_ssim_a_2k - best_ssim_b_2k
    delta_psnr_best_2k = best_psnr_a_2k - best_psnr_b_2k

    full_psnr_a_best, full_ssim_a_best, _ = evaluate_inversion(dec_a_best, client, test_full_loader, device)
    full_psnr_b_best, full_ssim_b_best, _ = evaluate_inversion(dec_b_best, client, test_full_loader, device)
    delta_ssim_full_best = full_ssim_a_best - full_ssim_b_best

    # 5. Tính toán độ chênh lệch theo mốc 10 Epochs cuối (Last 10)
    mean_a_ssim_10 = float(np.mean(last10_a_ssim)) if last10_a_ssim else best_a_ssim
    mean_b_ssim_10 = float(np.mean(last10_b_ssim)) if last10_b_ssim else best_b_ssim
    delta_ssim_last10 = mean_a_ssim_10 - mean_b_ssim_10

    mean_a_psnr_10 = float(np.mean(last10_a_psnr)) if last10_a_psnr else best_a_psnr
    mean_b_psnr_10 = float(np.mean(last10_b_psnr)) if last10_b_psnr else best_b_psnr
    delta_psnr_last10 = mean_a_psnr_10 - mean_b_psnr_10

    lpips_fn = get_lpips_fn(device=device)
    _, _, lpips_a_best = evaluate_inversion(dec_a_best, client, test_eval_loader, device, lpips_fn)
    _, _, lpips_b_best = evaluate_inversion(dec_b_best, client, test_eval_loader, device, lpips_fn)

    # Đánh giá nhận định dựa trên cả mốc Last10 và mốc Best
    primary_delta = delta_ssim_best_2k if abs(delta_ssim_best_2k) > abs(delta_ssim_last10) else delta_ssim_last10
    if abs(primary_delta) <= 0.05:
        verdict = f"Bất biến với Co-adaptation (Chênh lệch không đáng kể <= 0.05, Δ_best={delta_ssim_best_2k:+.4f}, Δ_last10={delta_ssim_last10:+.4f})"
    elif primary_delta > 0.05:
        verdict = f"Nhạy cảm với Co-adaptation (Co-adaptation tăng SSIM +{primary_delta:.4f})"
    else:
        verdict = f"Kháng Co-adaptation vượt trội (Co-adaptation thấp hơn Fresh {primary_delta:.4f})"

    # Lưu hình ảnh so sánh cả Best và Final
    grid_best_path = os.path.join(args.output_dir, f"grid_coadapted_{scenario_id}_best.png")
    save_comparison_grid(client, dec_a_best, dec_b_best, test_eval_loader, device, grid_best_path, num_images=8)

    grid_final_path = os.path.join(args.output_dir, f"grid_coadapted_{scenario_id}_final.png")
    save_comparison_grid(client, dec_a, dec_b, test_eval_loader, device, grid_final_path, num_images=8)

    # Vẽ biểu đồ
    plot_path = os.path.join(args.output_dir, f"curve_coadapted_{scenario_id}.png")
    plt.figure(figsize=(10, 4.5))
    ep_a = [h["epoch"] for h in history_a]
    ssim_a = [h["ssim"] for h in history_a]
    ep_b = [h["epoch_equiv"] for h in history_b]
    ssim_b = [h["ssim"] for h in history_b]
    plt.plot(ep_a, ssim_a, "b-o", label="Nhánh A: Co-adapted (Online)")
    plt.plot(ep_b, ssim_b, "r--s", label="Nhánh B: Fresh Matched Steps")
    plt.axhline(0.30, color="gray", linestyle=":", label="Ngưỡng Privacy (SSIM < 0.3)")
    plt.xlabel("Epochs (hoặc Epoch tương đương)")
    plt.ylabel("SSIM")
    plt.title(f"So Sánh Co-adapted vs Fresh trên AR-TAPE v0 (N_aux={args.aux_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    result = {
        "scenario_id": scenario_id,
        "k": args.k,
        "unprojector_mode": args.unprojector_mode,
        "fixed_bottleneck": getattr(args, "fixed_bottleneck", False),
        "lambda": args.lambda_val,
        "aux_size": args.aux_size,
        "seed": args.seed,
        "best_a_ssim_2k": round(best_ssim_a_2k, 4),
        "best_b_ssim_2k": round(best_ssim_b_2k, 4),
        "delta_ssim_best_2k": round(delta_ssim_best_2k, 4),
        "full_a_ssim_best": round(full_ssim_a_best, 4),
        "full_b_ssim_best": round(full_ssim_b_best, 4),
        "delta_ssim_full_best": round(delta_ssim_full_best, 4),
        "mean_a_ssim_10": round(mean_a_ssim_10, 4),
        "mean_b_ssim_10": round(mean_b_ssim_10, 4),
        "delta_ssim_last10": round(delta_ssim_last10, 4),
        "best_a_psnr_2k": round(best_psnr_a_2k, 2),
        "best_b_psnr_2k": round(best_psnr_b_2k, 2),
        "delta_psnr_best_2k": round(delta_psnr_best_2k, 2),
        "mean_a_psnr_10": round(mean_a_psnr_10, 2),
        "mean_b_psnr_10": round(mean_b_psnr_10, 2),
        "delta_psnr_last10": round(delta_psnr_last10, 2),
        "lpips_a_best": round(lpips_a_best, 4) if lpips_a_best else None,
        "lpips_b_best": round(lpips_b_best, 4) if lpips_b_best else None,
        "total_steps": total_steps_a,
        "verdict": verdict,
        "history_a": history_a,
        "history_b": history_b
    }

    print("\n" + "=" * 90)
    print(f"KẾT QUẢ ĐỐI CHUẨN CO-ADAPTED TRÊN AR-TAPE v0 ({scenario_id.upper()}):")
    print(f"  [CHECKPOINT TỐT NHẤT - PEAK SSIM]:")
    print(f"    - Nhánh A Best SSIM (2k / full 10k): {best_ssim_a_2k:.4f} / {full_ssim_a_best:.4f} (PSNR: {best_psnr_a_2k:.2f} dB, Ep {best_a_ep})")
    print(f"    - Nhánh B Best SSIM (2k / full 10k): {best_ssim_b_2k:.4f} / {full_ssim_b_best:.4f} (PSNR: {best_psnr_b_2k:.2f} dB)")
    print(f"    -> ΔSSIM Best (A - B) = {delta_ssim_best_2k:+.4f} (Full 10k: {delta_ssim_full_best:+.4f})")
    print(f"  [10 EPOCHS CUỐI - LAST 10 MEAN]:")
    print(f"    - Nhánh A Last10 SSIM: {mean_a_ssim_10:.4f} | PSNR: {mean_a_psnr_10:.2f} dB")
    print(f"    - Nhánh B Last10 SSIM: {mean_b_ssim_10:.4f} | PSNR: {mean_b_psnr_10:.2f} dB")
    print(f"    -> ΔSSIM Last10 (A - B) = {delta_ssim_last10:+.4f}")
    print(f"  - Nhận định: {verdict}")
    print("=" * 90 + "\n", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Co-adapted Attacker Evaluation trên AR-TAPE v0")
    parser.add_argument("--k", type=int, default=1024, help="Số chiều bottleneck k (mặc định: 1024)")
    parser.add_argument("--m", type=int, default=512, help="Số chiều ẩn MLP unprojector (mặc định: 512)")
    parser.add_argument("--unprojector-mode", type=str, default="linear_fixed", choices=["linear_fixed", "linear_learned", "mlp"], help="Chế độ unprojector tại server (mặc định: linear_fixed)")
    parser.add_argument("--fixed-bottleneck", action="store_true", help="Cố định ma trận chiếu trực giao R tại client (Mode F style)")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Số epoch warmup lambda từ 0 (mặc định: 0)")
    parser.add_argument("--lambda-val", type=float, default=0.1, help="Giá trị lambda (mặc định: 0.1)")
    parser.add_argument("--cap", type=float, default=1.5586, help="Cận trên chặn MSE loss (mặc định: 1.5586)")
    parser.add_argument("--aux-size", type=int, default=10000, help="Kích thước tập ảnh phụ D_aux (mặc định: 10000)")
    parser.add_argument("--test-eval-size", type=int, default=2000, help="Kích thước tập test nhanh (mặc định: 2000)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs Split Learning (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate SGD cho SL (mặc định: 0.1)")
    parser.add_argument("--decoder-lr", type=float, default=1e-3, help="Learning rate Adam cho Decoder (mặc định: 1e-3)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá mỗi N epochs (mặc định: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (mặc định: 42)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_v0_CoAdapted"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    result = run_coadapted_single(args, device)

    json_path = os.path.join(args.output_dir, f"results_coadapted_{result['scenario_id']}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output_dir, f"results_coadapted_{result['scenario_id']}.csv")
    keys = [k for k in result.keys() if not isinstance(result[k], (list, dict))]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow({k: result[k] for k in keys})

    print(f"[HOÀN TẤT] Kết quả Co-adapted đã lưu tại: {json_path} và {csv_path}")


if __name__ == "__main__":
    main()
