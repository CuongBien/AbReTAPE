#!/usr/bin/env python3
# Bước 3: Huấn luyện và Đánh giá các Baseline phòng thủ: B1 (Gaussian Noise), B2 (DP-SGD), B3 (NoPeek dCor)
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
from src.defenses import GaussianNoise, DPSGDClientOptimizer, compute_dp_epsilon, NoPeekDefense
from src.training import train_sl_epoch, evaluate_sl, EarlyStopping
from src.attacks import Decoder, train_inversion_epoch, evaluate_inversion
from src.metrics import get_lpips_fn, distance_correlation
from src.utils import plot_tradeoff_curves, save_reconstruction_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 3: Đánh giá Baseline B1 (Gaussian Noise), B2 (DP-SGD), B3 (NoPeek)")
    parser.add_argument("--defense", choices=["b1", "b2", "b3", "all"], default="b3",
                        help="Lựa chọn baseline: 'b1' (Gaussian Noise), 'b2' (DP-SGD), 'b3' (NoPeek dCor), hoặc 'all'")
    parser.add_argument("--epochs", type=int, default=10, help="Số epochs huấn luyện SL (mặc định: 10)")
    parser.add_argument("--attack-epochs", type=int, default=10, help="Số epochs huấn luyện Decoder tấn công (mặc định: 10)")
    parser.add_argument("--decoder-epochs", type=int, default=None, help="Alias cho --attack-epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate cho SL (mặc định: 0.05)")
    parser.add_argument("--decoder-lr", type=float, default=1e-3, help="Learning rate cho Decoder (mặc định: 1e-3)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá (mặc định: 5)")
    parser.add_argument("--patience", type=int, default=15, help="Số epochs chờ Early Stopping (mặc định: 15, 0 để tắt)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2,
                        help="Số luồng nạp dữ liệu (mặc định: 0 trên Windows để tránh crash IPC, 2 trên Linux)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Tự động khôi phục từ checkpoint gần nhất nếu có")
    parser.add_argument("--sweep", action="store_true", default=False, help="Chạy chế độ sweep toàn bộ dải tham số")

    # B1 params
    parser.add_argument("--sigma", type=float, default=None, help="Giá trị đơn lẻ sigma cho Gaussian Noise (B1)")
    parser.add_argument("--sigmas", type=str, default=None, help="Danh sách sigma ngăn cách bởi dấu phẩy (vd: 0.1,0.5,1.0)")

    # B2 params
    parser.add_argument("--sigma-dp", type=float, default=None, help="Giá trị đơn lẻ sigma_DP cho DP-SGD (B2)")
    parser.add_argument("--sigmas-dp", type=str, default=None, help="Danh sách sigma_DP ngăn cách bởi dấu phẩy (vd: 0.5,1.0,2.0)")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Max gradient clipping norm cho DP-SGD (B2)")
    parser.add_argument("--micro-batch-size", type=int, default=16, help="Kích thước micro-batch (tùy chọn)")

    # B3 params
    parser.add_argument("--alpha", type=float, default=None, help="Hệ số phạt dCor cho NoPeek (B3)")
    parser.add_argument("--alphas", type=str, default=None, help="Danh sách alpha ngăn cách bởi dấu phẩy (vd: 0.1,0.5,1.0)")

    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục dữ liệu")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output"), help="Thư mục lưu outputs")
    return parser.parse_args()


def resolve_output_dir(base_dir, suffix):
    if base_dir.rstrip("/\\").endswith(suffix):
        return base_dir
    return os.path.join(base_dir, suffix)


def save_defense_results(out_dir, defense_name, results, x_key, x_label):
    res_file = os.path.join(out_dir, f"results_{defense_name}.json")
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_file = os.path.join(out_dir, f"results_{defense_name}.csv")
    if results:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    plot_file = os.path.join(out_dir, f"{defense_name}_tradeoff_curves.png")
    try:
        plot_tradeoff_curves(results, save_path=plot_file, x_key=x_key, x_label=x_label)
    except Exception as e:
        print(f"[WARN] Không thể vẽ biểu đồ tradeoff: {e}")


def run_b1_gaussian(args, device):
    out_dir = resolve_output_dir(args.output_dir, "AbReTAPE_Step3")
    os.makedirs(out_dir, exist_ok=True)

    if args.sigma is not None and not args.sweep:
        sigmas = [args.sigma]
    elif args.sigmas is not None:
        sigmas = [float(s.strip()) for s in args.sigmas.split(",")]
    else:
        sigmas = [0.1, 0.5, 1.0]

    attack_epochs = args.decoder_epochs if args.decoder_epochs is not None else args.attack_epochs
    results = []

    res_file = os.path.join(out_dir, "results_b1.json")
    if args.resume and os.path.isfile(res_file):
        try:
            with open(res_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []

    print("\n" + "=" * 70)
    print("BƯỚC 3 — BASELINE B1: GAUSSIAN NOISE AT CUT LAYER")
    print(f"Thử nghiệm với các mức sigma: {sigmas} | SL Epochs: {args.epochs} | Attack Epochs: {attack_epochs}")
    print(f"LR Scheduler: CosineAnnealingLR | Early Stopping Patience: {args.patience}")
    print(f"Thư mục lưu kết quả: {out_dir}")
    print("=" * 70)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    lpips_fn = get_lpips_fn(device=device)

    for sigma in sigmas:
        recons_path = os.path.join(out_dir, f"b1_reconstruction_comparison_sigma{sigma}.png")
        completed_sigmas = [r.get("sigma") for r in results if "sigma" in r and "psnr" in r]
        if args.resume and sigma in completed_sigmas and os.path.isfile(recons_path):
            print(f"\n[RESUME] Baseline B1 với sigma = {sigma} đã hoàn thành toàn bộ trước đó. Bỏ qua.")
            continue

        print(f"\n---> ĐANG CHẠY BASELINE B1 VỚI SIGMA = {sigma} <---")
        client = ClientModel().to(device)
        server = ServerModel().to(device)
        noise = GaussianNoise(sigma=sigma).to(device)

        opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sched_c = CosineAnnealingLR(opt_c, T_max=args.epochs)
        sched_s = CosineAnnealingLR(opt_s, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()
        early_stopping_sl = EarlyStopping(patience=args.patience, mode="max")

        ckpt_last = os.path.join(out_dir, f"b1_last_sigma{sigma}.pt")
        ckpt_best = os.path.join(out_dir, f"b1_best_sigma{sigma}.pt")
        start_epoch = 1
        best_sl_acc = 0.0

        if args.resume and os.path.isfile(ckpt_last):
            ckpt = torch.load(ckpt_last, map_location=device)
            client.load_state_dict(ckpt["client"])
            server.load_state_dict(ckpt["server"])
            opt_c.load_state_dict(ckpt["opt_c"])
            opt_s.load_state_dict(ckpt["opt_s"])
            sched_c.load_state_dict(ckpt["sched_c"])
            sched_s.load_state_dict(ckpt["sched_s"])
            start_epoch = ckpt["epoch"] + 1
            best_sl_acc = ckpt.get("best_sl_acc", 0.0)
            early_stopping_sl.best_score = best_sl_acc
            print(f"[RESUME] Đã khôi phục SL B1 (σ={sigma}) từ epoch {start_epoch-1} (Best Acc: {best_sl_acc*100:.2f}%)")

        if start_epoch <= args.epochs:
            for epoch in range(start_epoch, args.epochs + 1):
                train_loss, train_acc = train_sl_epoch(client, server, trainloader, opt_c, opt_s, criterion, device, defense=noise)
                sched_c.step()
                sched_s.step()

                if epoch % args.eval_freq == 0 or epoch == args.epochs:
                    _, t_acc = evaluate_sl(client, server, testloader, device, defense=noise, criterion=criterion)
                    if t_acc > best_sl_acc:
                        best_sl_acc = t_acc
                        torch.save({"client": client.state_dict(), "server": server.state_dict(), "best_sl_acc": best_sl_acc, "epoch": epoch}, ckpt_best)
                    print(f"[SL Train σ={sigma}] Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {t_acc*100:.2f}% (Best: {best_sl_acc*100:.2f}%) | LR: {opt_c.param_groups[0]['lr']:.4f}", flush=True)
                    torch.save({
                        "epoch": epoch,
                        "client": client.state_dict(),
                        "server": server.state_dict(),
                        "opt_c": opt_c.state_dict(),
                        "opt_s": opt_s.state_dict(),
                        "sched_c": sched_c.state_dict(),
                        "sched_s": sched_s.state_dict(),
                        "best_sl_acc": best_sl_acc,
                    }, ckpt_last)
                    if early_stopping_sl.step(t_acc, epoch=epoch):
                        print(f"\n[EARLY STOPPING] Dừng sớm SL tại epoch {epoch} do test acc không cải thiện sau {args.patience} lần đánh giá! Best Test Acc: {best_sl_acc*100:.2f}%")
                        break
                else:
                    print(f"[SL Train σ={sigma}] Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | LR: {opt_c.param_groups[0]['lr']:.4f}", flush=True)
                    torch.save({
                        "epoch": epoch,
                        "client": client.state_dict(),
                        "server": server.state_dict(),
                        "opt_c": opt_c.state_dict(),
                        "opt_s": opt_s.state_dict(),
                        "sched_c": sched_c.state_dict(),
                        "sched_s": sched_s.state_dict(),
                        "best_sl_acc": best_sl_acc,
                    }, ckpt_last)

        if os.path.isfile(ckpt_best):
            best_dict = torch.load(ckpt_best, map_location=device)
            client.load_state_dict(best_dict["client"])
            server.load_state_dict(best_dict["server"])

        _, test_acc = evaluate_sl(client, server, testloader, device, defense=noise, criterion=criterion)
        print(f"[SL Result σ={sigma}] Final Test Accuracy: {test_acc*100:.2f}% (Best: {best_sl_acc*100:.2f}%)")

        print(f"[Attack] Đang huấn luyện Decoder tấn công đối phó với σ = {sigma}...")
        decoder = Decoder().to(device)
        opt_d = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr, weight_decay=1e-5)
        sched_d = CosineAnnealingLR(opt_d, T_max=attack_epochs)
        crit_d = nn.MSELoss()
        early_stopping_d = EarlyStopping(patience=args.patience, mode="max")

        ckpt_dec_last = os.path.join(out_dir, f"b1_dec_last_sigma{sigma}.pt")
        ckpt_dec_best = os.path.join(out_dir, f"b1_dec_best_sigma{sigma}.pt")
        start_dec_ep = 1
        best_attack_psnr = 0.0

        if args.resume and os.path.isfile(ckpt_dec_last):
            dec_ckpt = torch.load(ckpt_dec_last, map_location=device)
            decoder.load_state_dict(dec_ckpt["decoder"])
            opt_d.load_state_dict(dec_ckpt["opt_d"])
            sched_d.load_state_dict(dec_ckpt["sched_d"])
            start_dec_ep = dec_ckpt["epoch"] + 1
            best_attack_psnr = dec_ckpt.get("best_psnr", 0.0)
            early_stopping_d.best_score = best_attack_psnr
            print(f"[RESUME] Đã khôi phục Decoder B1 (σ={sigma}) từ epoch {start_dec_ep-1} (Best PSNR: {best_attack_psnr:.2f} dB)")

        if start_dec_ep <= attack_epochs:
            for ep in range(start_dec_ep, attack_epochs + 1):
                train_inversion_epoch(client, decoder, trainloader, opt_d, crit_d, device, defense=noise)
                sched_d.step()
                if ep % args.eval_freq == 0 or ep == attack_epochs:
                    _, cur_psnr, _, _ = evaluate_inversion(
                        client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, defense=noise, criterion=crit_d, lpips_fn=None
                    )
                    if cur_psnr is not None and cur_psnr > best_attack_psnr:
                        best_attack_psnr = cur_psnr
                        torch.save({"decoder": decoder.state_dict(), "best_psnr": best_attack_psnr, "epoch": ep}, ckpt_dec_best)
                    print(f"[Attack σ={sigma}] Epoch {ep:2d}/{attack_epochs} | PSNR: {cur_psnr:.2f} dB (Best: {best_attack_psnr:.2f} dB)", flush=True)
                    torch.save({
                        "epoch": ep,
                        "decoder": decoder.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "sched_d": sched_d.state_dict(),
                        "best_psnr": best_attack_psnr,
                    }, ckpt_dec_last)
                    if cur_psnr is not None and early_stopping_d.step(cur_psnr, epoch=ep):
                        print(f"[EARLY STOPPING] Dừng sớm Decoder tại epoch {ep} do PSNR không cải thiện sau {args.patience} lần đánh giá!")
                        break
                else:
                    torch.save({
                        "epoch": ep,
                        "decoder": decoder.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "sched_d": sched_d.state_dict(),
                        "best_psnr": best_attack_psnr,
                    }, ckpt_dec_last)

        if os.path.isfile(ckpt_dec_best):
            best_dec_dict = torch.load(ckpt_dec_best, map_location=device)
            decoder.load_state_dict(best_dec_dict["decoder"])

        mse, psnr, ssim, lpips_val = evaluate_inversion(
            client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, defense=noise, criterion=crit_d, lpips_fn=lpips_fn
        )
        print(f"[Security σ={sigma}] PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | LPIPS: {lpips_val if lpips_val else 'N/A'}")

        # Lưu ảnh tái tạo
        save_reconstruction_grid(client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=recons_path, num_images=8, defense=noise)
        default_recons = os.path.join(out_dir, "b1_reconstruction_comparison.png")
        save_reconstruction_grid(client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=default_recons, num_images=8, defense=noise)

        cur_entry = {
            "sigma": sigma,
            "test_acc": test_acc,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_val
        }
        results = [r for r in results if r.get("sigma") != sigma]
        results.append(cur_entry)
        results.sort(key=lambda r: r.get("sigma", 0.0))
        save_defense_results(out_dir, "b1", results, x_key="sigma", x_label="Cường độ Nhiễu Gauss (sigma)")

    print(f"\n[DONE] Hoàn thành Baseline B1! Kết quả lưu tại: {res_file}")


def run_b2_dpsgd(args, device):
    out_dir = resolve_output_dir(args.output_dir, "AbReTAPE_Step3_B2")
    os.makedirs(out_dir, exist_ok=True)

    if args.sigma_dp is not None and not args.sweep:
        sigmas_dp = [args.sigma_dp]
    elif args.sigmas_dp is not None:
        sigmas_dp = [float(s.strip()) for s in args.sigmas_dp.split(",")]
    else:
        sigmas_dp = [0.5, 1.0, 2.0]

    attack_epochs = args.decoder_epochs if args.decoder_epochs is not None else args.attack_epochs
    results = []

    res_file = os.path.join(out_dir, "results_b2.json")
    if args.resume and os.path.isfile(res_file):
        try:
            with open(res_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []

    print("\n" + "=" * 70)
    print("BƯỚC 3 — BASELINE B2: DP-SGD CLIENT OPTIMIZER")
    print(f"Thử nghiệm với các mức sigma_DP: {sigmas_dp} | SL Epochs: {args.epochs} | Attack Epochs: {attack_epochs}")
    print(f"LR Scheduler: CosineAnnealingLR | Early Stopping Patience: {args.patience}")
    print(f"Thư mục lưu kết quả: {out_dir}")
    print("=" * 70)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    lpips_fn = get_lpips_fn(device=device)

    for s_dp in sigmas_dp:
        eps = compute_dp_epsilon(epochs=args.epochs, batch_size=args.batch_size, noise_multiplier=s_dp)
        recons_path = os.path.join(out_dir, f"b2_reconstruction_comparison_sigma{s_dp}.png")
        completed_sigmas = [r.get("sigma_dp") for r in results if "sigma_dp" in r and "psnr" in r]
        if args.resume and s_dp in completed_sigmas and os.path.isfile(recons_path):
            print(f"\n[RESUME] Baseline B2 với sigma_DP = {s_dp} đã hoàn thành toàn bộ trước đó. Bỏ qua.")
            continue

        print(f"\n---> ĐANG CHẠY BASELINE B2: σ_DP = {s_dp} (Epsilon: {eps:.2f}, Delta: 1e-5) <---")

        client = ClientModel().to(device)
        server = ServerModel().to(device)

        base_opt = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        dp_opt = DPSGDClientOptimizer(client, base_opt, max_grad_norm=args.clip_norm, noise_multiplier=s_dp)
        opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sched_c = CosineAnnealingLR(base_opt, T_max=args.epochs)
        sched_s = CosineAnnealingLR(opt_s, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()
        early_stopping_sl = EarlyStopping(patience=args.patience, mode="max")

        ckpt_last = os.path.join(out_dir, f"b2_last_sigma{s_dp}.pt")
        ckpt_best = os.path.join(out_dir, f"b2_best_sigma{s_dp}.pt")
        start_epoch = 1
        best_sl_acc = 0.0

        if args.resume and os.path.isfile(ckpt_last):
            ckpt = torch.load(ckpt_last, map_location=device)
            client.load_state_dict(ckpt["client"])
            server.load_state_dict(ckpt["server"])
            base_opt.load_state_dict(ckpt["opt_c"])
            opt_s.load_state_dict(ckpt["opt_s"])
            sched_c.load_state_dict(ckpt["sched_c"])
            sched_s.load_state_dict(ckpt["sched_s"])
            start_epoch = ckpt["epoch"] + 1
            best_sl_acc = ckpt.get("best_sl_acc", 0.0)
            early_stopping_sl.best_score = best_sl_acc
            print(f"[RESUME] Đã khôi phục SL B2 (σ_DP={s_dp}) từ epoch {start_epoch-1} (Best Acc: {best_sl_acc*100:.2f}%)")

        if start_epoch <= args.epochs:
            for epoch in range(start_epoch, args.epochs + 1):
                train_loss, train_acc = train_sl_epoch(client, server, trainloader, dp_opt, opt_s, criterion, device, client_opt_is_dpsgd=True)
                sched_c.step()
                sched_s.step()

                if epoch % args.eval_freq == 0 or epoch == args.epochs:
                    _, t_acc = evaluate_sl(client, server, testloader, device, criterion=criterion)
                    if t_acc > best_sl_acc:
                        best_sl_acc = t_acc
                        torch.save({"client": client.state_dict(), "server": server.state_dict(), "best_sl_acc": best_sl_acc, "epoch": epoch}, ckpt_best)
                    print(f"[DP-SL σ={s_dp}] Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {t_acc*100:.2f}% (Best: {best_sl_acc*100:.2f}%) | LR: {base_opt.param_groups[0]['lr']:.4f}", flush=True)
                    torch.save({
                        "epoch": epoch,
                        "client": client.state_dict(),
                        "server": server.state_dict(),
                        "opt_c": base_opt.state_dict(),
                        "opt_s": opt_s.state_dict(),
                        "sched_c": sched_c.state_dict(),
                        "sched_s": sched_s.state_dict(),
                        "best_sl_acc": best_sl_acc,
                    }, ckpt_last)
                    if early_stopping_sl.step(t_acc, epoch=epoch):
                        print(f"\n[EARLY STOPPING] Dừng sớm DP-SL tại epoch {epoch} do test acc không cải thiện sau {args.patience} lần đánh giá! Best: {best_sl_acc*100:.2f}%")
                        break
                else:
                    print(f"[DP-SL σ={s_dp}] Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | LR: {base_opt.param_groups[0]['lr']:.4f}", flush=True)
                    torch.save({
                        "epoch": epoch,
                        "client": client.state_dict(),
                        "server": server.state_dict(),
                        "opt_c": base_opt.state_dict(),
                        "opt_s": opt_s.state_dict(),
                        "sched_c": sched_c.state_dict(),
                        "sched_s": sched_s.state_dict(),
                        "best_sl_acc": best_sl_acc,
                    }, ckpt_last)

        if os.path.isfile(ckpt_best):
            best_dict = torch.load(ckpt_best, map_location=device)
            client.load_state_dict(best_dict["client"])
            server.load_state_dict(best_dict["server"])

        _, test_acc = evaluate_sl(client, server, testloader, device, criterion=criterion)
        print(f"[DP-SL Result σ={s_dp}] Final Test Accuracy: {test_acc*100:.2f}% (Epsilon: {eps:.2f})")

        print(f"[Attack] Đang huấn luyện Decoder tấn công đối phó với DP-SGD (σ_DP={s_dp})...")
        decoder = Decoder().to(device)
        opt_d = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr, weight_decay=1e-5)
        sched_d = CosineAnnealingLR(opt_d, T_max=attack_epochs)
        crit_d = nn.MSELoss()
        early_stopping_d = EarlyStopping(patience=args.patience, mode="max")

        ckpt_dec_last = os.path.join(out_dir, f"b2_dec_last_sigma{s_dp}.pt")
        ckpt_dec_best = os.path.join(out_dir, f"b2_dec_best_sigma{s_dp}.pt")
        start_dec_ep = 1
        best_attack_psnr = 0.0

        if args.resume and os.path.isfile(ckpt_dec_last):
            dec_ckpt = torch.load(ckpt_dec_last, map_location=device)
            decoder.load_state_dict(dec_ckpt["decoder"])
            opt_d.load_state_dict(dec_ckpt["opt_d"])
            sched_d.load_state_dict(dec_ckpt["sched_d"])
            start_dec_ep = dec_ckpt["epoch"] + 1
            best_attack_psnr = dec_ckpt.get("best_psnr", 0.0)
            early_stopping_d.best_score = best_attack_psnr
            print(f"[RESUME] Đã khôi phục Decoder B2 (σ_DP={s_dp}) từ epoch {start_dec_ep-1} (Best PSNR: {best_attack_psnr:.2f} dB)")

        if start_dec_ep <= attack_epochs:
            for ep in range(start_dec_ep, attack_epochs + 1):
                train_inversion_epoch(client, decoder, trainloader, opt_d, crit_d, device)
                sched_d.step()
                if ep % args.eval_freq == 0 or ep == attack_epochs:
                    _, cur_psnr, _, _ = evaluate_inversion(
                        client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, criterion=crit_d, lpips_fn=None
                    )
                    if cur_psnr is not None and cur_psnr > best_attack_psnr:
                        best_attack_psnr = cur_psnr
                        torch.save({"decoder": decoder.state_dict(), "best_psnr": best_attack_psnr, "epoch": ep}, ckpt_dec_best)
                    print(f"[Attack σ_DP={s_dp}] Epoch {ep:2d}/{attack_epochs} | PSNR: {cur_psnr:.2f} dB (Best: {best_attack_psnr:.2f} dB)", flush=True)
                    torch.save({
                        "epoch": ep,
                        "decoder": decoder.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "sched_d": sched_d.state_dict(),
                        "best_psnr": best_attack_psnr,
                    }, ckpt_dec_last)
                    if cur_psnr is not None and early_stopping_d.step(cur_psnr, epoch=ep):
                        print(f"[EARLY STOPPING] Dừng sớm Decoder tại epoch {ep} do PSNR không cải thiện sau {args.patience} lần đánh giá!")
                        break
                else:
                    torch.save({
                        "epoch": ep,
                        "decoder": decoder.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "sched_d": sched_d.state_dict(),
                        "best_psnr": best_attack_psnr,
                    }, ckpt_dec_last)

        if os.path.isfile(ckpt_dec_best):
            best_dec_dict = torch.load(ckpt_dec_best, map_location=device)
            decoder.load_state_dict(best_dec_dict["decoder"])

        mse, psnr, ssim, lpips_val = evaluate_inversion(
            client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, criterion=crit_d, lpips_fn=lpips_fn
        )
        print(f"[Security DP σ={s_dp}] PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

        # Lưu ảnh tái tạo
        save_reconstruction_grid(client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=recons_path, num_images=8)
        default_recons = os.path.join(out_dir, "b2_reconstruction_comparison.png")
        save_reconstruction_grid(client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=default_recons, num_images=8)

        cur_entry = {
            "sigma_dp": s_dp,
            "epsilon": eps,
            "test_acc": test_acc,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_val
        }
        results = [r for r in results if r.get("sigma_dp") != s_dp]
        results.append(cur_entry)
        results.sort(key=lambda r: r.get("sigma_dp", 0.0))
        save_defense_results(out_dir, "b2", results, x_key="sigma_dp", x_label="Cường độ Nhiễu DP (sigma_DP)")

    print(f"\n[DONE] Hoàn thành Baseline B2! Kết quả lưu tại: {res_file}")


def run_b3_nopeek(args, device):
    out_dir = resolve_output_dir(args.output_dir, "AbReTAPE_Step3_B3")
    os.makedirs(out_dir, exist_ok=True)

    if args.alpha is not None and not args.sweep:
        alphas = [args.alpha]
    elif args.alphas is not None:
        alphas = [float(a.strip()) for a in args.alphas.split(",")]
    else:
        alphas = [0.1, 0.5, 1.0]

    attack_epochs = args.decoder_epochs if args.decoder_epochs is not None else args.attack_epochs
    results = []

    res_file = os.path.join(out_dir, "results_b3.json")
    if args.resume and os.path.isfile(res_file):
        try:
            with open(res_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []

    print("\n" + "=" * 70)
    print("BƯỚC 3 — BASELINE B3: NOPEEK (DISTANCE CORRELATION PENALTY)")
    print(f"Thử nghiệm với các mức alpha (dCor penalty): {alphas} | SL Epochs: {args.epochs} | Attack Epochs: {attack_epochs}")
    print(f"LR Scheduler: CosineAnnealingLR | Early Stopping Patience: {args.patience}")
    print(f"Thư mục lưu kết quả: {out_dir}")
    print("=" * 70)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    lpips_fn = get_lpips_fn(device=device)

    for alpha in alphas:
        recons_path = os.path.join(out_dir, f"b3_reconstruction_comparison_alpha{alpha}.png")
        completed_alphas = [r.get("alpha") for r in results if "alpha" in r and "psnr" in r]
        if args.resume and alpha in completed_alphas and os.path.isfile(recons_path):
            print(f"\n[RESUME] Baseline B3 với alpha = {alpha} đã hoàn thành toàn bộ trước đó. Bỏ qua.")
            continue

        print(f"\n---> ĐANG CHẠY BASELINE B3: ALPHA = {alpha} <---")
        client = ClientModel().to(device)
        server = ServerModel().to(device)
        nopeek = NoPeekDefense(alpha=alpha).to(device)

        opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sched_c = CosineAnnealingLR(opt_c, T_max=args.epochs)
        sched_s = CosineAnnealingLR(opt_s, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()
        early_stopping_sl = EarlyStopping(patience=args.patience, mode="max")

        ckpt_last = os.path.join(out_dir, f"b3_last_alpha{alpha}.pt")
        ckpt_best = os.path.join(out_dir, f"b3_best_alpha{alpha}.pt")
        start_epoch = 1
        best_sl_acc = 0.0

        if args.resume and os.path.isfile(ckpt_last):
            ckpt = torch.load(ckpt_last, map_location=device)
            client.load_state_dict(ckpt["client"])
            server.load_state_dict(ckpt["server"])
            opt_c.load_state_dict(ckpt["opt_c"])
            opt_s.load_state_dict(ckpt["opt_s"])
            sched_c.load_state_dict(ckpt["sched_c"])
            sched_s.load_state_dict(ckpt["sched_s"])
            start_epoch = ckpt["epoch"] + 1
            best_sl_acc = ckpt.get("best_sl_acc", 0.0)
            early_stopping_sl.best_score = best_sl_acc
            print(f"[RESUME] Đã khôi phục SL B3 (α={alpha}) từ epoch {start_epoch-1} (Best Acc: {best_sl_acc*100:.2f}%)")

        if start_epoch <= args.epochs:
            for epoch in range(start_epoch, args.epochs + 1):
                train_loss, train_acc = train_sl_epoch(client, server, trainloader, opt_c, opt_s, criterion, device, defense=nopeek)
                sched_c.step()
                sched_s.step()

                if epoch % args.eval_freq == 0 or epoch == args.epochs:
                    _, t_acc = evaluate_sl(client, server, testloader, device, criterion=criterion)
                    if t_acc > best_sl_acc:
                        best_sl_acc = t_acc
                        torch.save({"client": client.state_dict(), "server": server.state_dict(), "best_sl_acc": best_sl_acc, "epoch": epoch}, ckpt_best)
                    print(f"[NoPeek α={alpha}] Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {t_acc*100:.2f}% (Best: {best_sl_acc*100:.2f}%) | LR: {opt_c.param_groups[0]['lr']:.4f}", flush=True)
                    torch.save({
                        "epoch": epoch,
                        "client": client.state_dict(),
                        "server": server.state_dict(),
                        "opt_c": opt_c.state_dict(),
                        "opt_s": opt_s.state_dict(),
                        "sched_c": sched_c.state_dict(),
                        "sched_s": sched_s.state_dict(),
                        "best_sl_acc": best_sl_acc,
                    }, ckpt_last)
                    if early_stopping_sl.step(t_acc, epoch=epoch):
                        print(f"\n[EARLY STOPPING] Dừng sớm NoPeek tại epoch {epoch} do test acc không cải thiện sau {args.patience} lần đánh giá! Best: {best_sl_acc*100:.2f}%")
                        break
                else:
                    print(f"[NoPeek α={alpha}] Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | LR: {opt_c.param_groups[0]['lr']:.4f}", flush=True)
                    torch.save({
                        "epoch": epoch,
                        "client": client.state_dict(),
                        "server": server.state_dict(),
                        "opt_c": opt_c.state_dict(),
                        "opt_s": opt_s.state_dict(),
                        "sched_c": sched_c.state_dict(),
                        "sched_s": sched_s.state_dict(),
                        "best_sl_acc": best_sl_acc,
                    }, ckpt_last)

        if os.path.isfile(ckpt_best):
            best_dict = torch.load(ckpt_best, map_location=device)
            client.load_state_dict(best_dict["client"])
            server.load_state_dict(best_dict["server"])

        _, test_acc = evaluate_sl(client, server, testloader, device, criterion=criterion)

        # Đo dCor trung bình trên test set
        client.eval()
        dcor_sum = 0.0
        n_eval = 0
        with torch.no_grad():
            for x_t, _ in testloader:
                x_t = x_t.to(device)
                z_t = client(x_t)
                dcor_sum += distance_correlation(x_t, z_t).item()
                n_eval += 1
                if n_eval >= 15:
                    break
        avg_dcor = dcor_sum / max(n_eval, 1)
        print(f"[NoPeek Result α={alpha}] Test Acc: {test_acc*100:.2f}% | dCor(X, Z'): {avg_dcor:.4f}")

        # Đánh giá Feature Inversion Attack
        print(f"[Attack] Đang huấn luyện Decoder tái tạo đối phó với NoPeek (α={alpha})...")
        decoder = Decoder().to(device)
        opt_d = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr, weight_decay=1e-5)
        sched_d = CosineAnnealingLR(opt_d, T_max=attack_epochs)
        crit_d = nn.MSELoss()
        early_stopping_d = EarlyStopping(patience=args.patience, mode="max")

        ckpt_dec_last = os.path.join(out_dir, f"b3_dec_last_alpha{alpha}.pt")
        ckpt_dec_best = os.path.join(out_dir, f"b3_dec_best_alpha{alpha}.pt")
        start_dec_ep = 1
        best_attack_psnr = 0.0

        if args.resume and os.path.isfile(ckpt_dec_last):
            dec_ckpt = torch.load(ckpt_dec_last, map_location=device)
            decoder.load_state_dict(dec_ckpt["decoder"])
            opt_d.load_state_dict(dec_ckpt["opt_d"])
            sched_d.load_state_dict(dec_ckpt["sched_d"])
            start_dec_ep = dec_ckpt["epoch"] + 1
            best_attack_psnr = dec_ckpt.get("best_psnr", 0.0)
            early_stopping_d.best_score = best_attack_psnr
            print(f"[RESUME] Đã khôi phục Decoder B3 (α={alpha}) từ epoch {start_dec_ep-1} (Best PSNR: {best_attack_psnr:.2f} dB)")

        if start_dec_ep <= attack_epochs:
            for ep in range(start_dec_ep, attack_epochs + 1):
                train_inversion_epoch(client, decoder, trainloader, opt_d, crit_d, device)
                sched_d.step()
                if ep % args.eval_freq == 0 or ep == attack_epochs:
                    _, cur_psnr, _, _ = evaluate_inversion(
                        client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, criterion=crit_d, lpips_fn=None
                    )
                    if cur_psnr is not None and cur_psnr > best_attack_psnr:
                        best_attack_psnr = cur_psnr
                        torch.save({"decoder": decoder.state_dict(), "best_psnr": best_attack_psnr, "epoch": ep}, ckpt_dec_best)
                    print(f"[Attack α={alpha}] Epoch {ep:2d}/{attack_epochs} | PSNR: {cur_psnr:.2f} dB (Best: {best_attack_psnr:.2f} dB)", flush=True)
                    torch.save({
                        "epoch": ep,
                        "decoder": decoder.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "sched_d": sched_d.state_dict(),
                        "best_psnr": best_attack_psnr,
                    }, ckpt_dec_last)
                    if cur_psnr is not None and early_stopping_d.step(cur_psnr, epoch=ep):
                        print(f"[EARLY STOPPING] Dừng sớm Decoder tại epoch {ep} do PSNR không cải thiện sau {args.patience} lần đánh giá!")
                        break
                else:
                    torch.save({
                        "epoch": ep,
                        "decoder": decoder.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "sched_d": sched_d.state_dict(),
                        "best_psnr": best_attack_psnr,
                    }, ckpt_dec_last)

        if os.path.isfile(ckpt_dec_best):
            best_dec_dict = torch.load(ckpt_dec_best, map_location=device)
            decoder.load_state_dict(best_dec_dict["decoder"])

        mse, psnr, ssim, lpips_val = evaluate_inversion(
            client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, criterion=crit_d, lpips_fn=lpips_fn
        )
        print(f"[Security NoPeek α={alpha}] PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | LPIPS: {lpips_val if lpips_val else 'N/A'}")

        # Lưu ảnh tái tạo
        save_reconstruction_grid(client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=recons_path, num_images=8)
        default_recons = os.path.join(out_dir, "b3_reconstruction_comparison.png")
        save_reconstruction_grid(client, decoder, testloader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=default_recons, num_images=8)

        cur_entry = {
            "alpha": alpha,
            "dcor": avg_dcor,
            "test_acc": test_acc,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_val
        }
        results = [r for r in results if r.get("alpha") != alpha]
        results.append(cur_entry)
        results.sort(key=lambda r: r.get("alpha", 0.0))
        save_defense_results(out_dir, "b3", results, x_key="alpha", x_label="Hệ số phạt NoPeek (alpha)")

    print(f"\n[DONE] Hoàn thành Baseline B3 (NoPeek)! Kết quả lưu tại: {res_file}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.defense in ["b1", "all"]:
        run_b1_gaussian(args, device)
    if args.defense in ["b2", "all"]:
        run_b2_dpsgd(args, device)
    if args.defense in ["b3", "all"]:
        run_b3_nopeek(args, device)


if __name__ == "__main__":
    main()
