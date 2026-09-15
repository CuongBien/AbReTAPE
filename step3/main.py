# Bước 3 — Baseline 1 (B1): Phòng thủ Nhiễu Gaussian tại Cut Layer và Tấn công Tái tạo Thích ứng
import os
import sys
import time
import json
import csv
import argparse

# Thiết lập UTF-8 encoding cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP0_DIR = os.path.join(PROJECT_ROOT, "step0")
STEP1_DIR = os.path.join(PROJECT_ROOT, "step1")

for path in reversed([SCRIPT_DIR, STEP0_DIR, STEP1_DIR, PROJECT_ROOT]):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import torch
import torch.nn as nn
from model import ClientModel, ServerModel
from data import get_cifar10
from decoder import Decoder
from metrics import get_lpips_fn
from defense import GaussianNoise
from train_sl import train_epoch, evaluate_sl
from attack import train_adaptive_decoder_epoch, evaluate_attack
from plot_b1 import plot_tradeoff, plot_reconstruction_grid

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def run_pipeline_for_sigma(sigma, args, trainloader, testloader, device, lpips_fn=None):
    """
    Chạy toàn bộ pipeline thống nhất cho 1 giá trị sigma:
    1. Train Split Learning có nhiễu Gaussian (epochs epoch).
    2. Đóng băng Client & phòng thủ.
    3. Train Decoder tấn công tái tạo thích ứng trên IR nhiễu (decoder_epochs epoch).
    4. Đo lường Test Accuracy (Utility) và PSNR, SSIM, LPIPS (Security).
    """
    print(f"\n==================================================", flush=True)
    print(f"   BƯỚC 3 (B1): HUẤN LUYỆN VỚI SIGMA = {sigma}", flush=True)
    print(f"==================================================", flush=True)

    ckpt_path = os.path.join(args.output_dir, f"b1_sigma_{sigma}.pt")
    dec_ckpt_path = os.path.join(args.output_dir, f"decoder_b1_sigma_{sigma}.pt")
    history_path = os.path.join(args.output_dir, f"history_sigma_{sigma}.json")

    # 1. Khởi tạo mô hình
    client = ClientModel().to(device)
    server = ServerModel().to(device)
    defense = GaussianNoise(sigma=sigma, apply_on_eval=True).to(device)

    opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched_c = torch.optim.lr_scheduler.MultiStepLR(opt_c, milestones=[50, 75], gamma=0.1)
    sched_s = torch.optim.lr_scheduler.MultiStepLR(opt_s, milestones=[50, 75], gamma=0.1)
    criterion_task = nn.CrossEntropyLoss()

    start_epoch = 1
    best_acc = 0.0
    history_sl = []

    # Kiểm tra resume nếu có
    if args.resume and os.path.isfile(ckpt_path):
        print(f"[RESUME] Nạp checkpoint SL từ: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        client.load_state_dict(ckpt["client"])
        server.load_state_dict(ckpt["server"])
        if "opt_c" in ckpt:
            opt_c.load_state_dict(ckpt["opt_c"])
        if "opt_s" in ckpt:
            opt_s.load_state_dict(ckpt["opt_s"])
        if "sched_c" in ckpt:
            sched_c.load_state_dict(ckpt["sched_c"])
        if "sched_s" in ckpt:
            sched_s.load_state_dict(ckpt["sched_s"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        history_sl = ckpt.get("history_sl", [])

    # Huấn luyện Split Learning
    if start_epoch <= args.epochs:
        print(f"\n--- PHA 1: Huấn luyện Split Learning có Nhiễu Gaussian (σ={sigma}) ---", flush=True)
        t_sl_start = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            loss_train, acc_train = train_epoch(client, server, defense, trainloader, opt_c, opt_s, criterion_task, device)
            curr_lr = opt_c.param_groups[0]["lr"]
            sched_c.step()
            sched_s.step()
            epoch_time = time.time() - t0

            is_eval = (epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs)
            loss_val, acc_val = (None, None)
            if is_eval:
                loss_val, acc_val = evaluate_sl(client, server, defense, testloader, device, criterion=criterion_task)
                if acc_val > best_acc:
                    best_acc = acc_val
                print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {loss_train:.4f} | Train Acc: {acc_train*100:.2f}% | Test Acc: {acc_val*100:.2f}% | LR: {curr_lr:.4f} | Time: {epoch_time:.1f}s", flush=True)
            else:
                print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {loss_train:.4f} | Train Acc: {acc_train*100:.2f}% | LR: {curr_lr:.4f} | Time: {epoch_time:.1f}s", flush=True)

            history_sl.append({
                "epoch": epoch,
                "loss_train": float(loss_train),
                "acc_train": float(acc_train),
                "loss_val": float(loss_val) if loss_val is not None else None,
                "acc_val": float(acc_val) if acc_val is not None else None,
            })

            # Lưu checkpoint ngắt quãng
            torch.save({
                "epoch": epoch,
                "sigma": sigma,
                "client": client.state_dict(),
                "server": server.state_dict(),
                "opt_c": opt_c.state_dict(),
                "opt_s": opt_s.state_dict(),
                "sched_c": sched_c.state_dict(),
                "sched_s": sched_s.state_dict(),
                "best_acc": best_acc,
                "history_sl": history_sl,
            }, ckpt_path)

        print(f"[DONE] Huấn luyện SL (σ={sigma}) hoàn thành trong {(time.time() - t_sl_start)/60:.2f} phút. Best Acc: {best_acc*100:.2f}%", flush=True)
    else:
        print(f"[INFO] Bỏ qua Pha 1 vì đã train đủ {args.epochs} epochs.", flush=True)

    # Đo độ chính xác phân loại cuối cùng
    _, final_test_acc = evaluate_sl(client, server, defense, testloader, device, criterion=criterion_task)

    # 2. Huấn luyện Decoder tấn công tái tạo thích ứng
    print(f"\n--- PHA 2: Tấn công Tái tạo Thích ứng (Decoder Attack, σ={sigma}) ---", flush=True)
    decoder = Decoder().to(device)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    criterion_recon = nn.MSELoss()

    dec_start_epoch = 1
    history_dec = []
    if args.resume and os.path.isfile(dec_ckpt_path):
        print(f"[RESUME] Nạp checkpoint Decoder từ: {dec_ckpt_path}", flush=True)
        ckpt_dec = torch.load(dec_ckpt_path, map_location=device)
        decoder.load_state_dict(ckpt_dec["decoder"])
        opt_d.load_state_dict(ckpt_dec["opt_d"])
        dec_start_epoch = ckpt_dec.get("epoch", 0) + 1
        history_dec = ckpt_dec.get("history_dec", [])

    if dec_start_epoch <= args.decoder_epochs:
        t_dec_start = time.time()
        for d_epoch in range(dec_start_epoch, args.decoder_epochs + 1):
            t0 = time.time()
            loss_dec = train_adaptive_decoder_epoch(client, decoder, defense, trainloader, opt_d, criterion_recon, device)
            d_time = time.time() - t0

            is_eval_dec = (d_epoch % 5 == 0 or d_epoch == 1 or d_epoch == args.decoder_epochs)
            if is_eval_dec:
                print(f"Decoder Epoch {d_epoch:2d}/{args.decoder_epochs} | Recon Loss (MSE): {loss_dec:.5f} | Time: {d_time:.1f}s", flush=True)
            history_dec.append({"epoch": d_epoch, "loss": float(loss_dec)})

            torch.save({
                "epoch": d_epoch,
                "sigma": sigma,
                "decoder": decoder.state_dict(),
                "opt_d": opt_d.state_dict(),
                "history_dec": history_dec,
            }, dec_ckpt_path)

        print(f"[DONE] Huấn luyện Decoder (σ={sigma}) hoàn thành trong {(time.time() - t_dec_start)/60:.2f} phút.", flush=True)
    else:
        print(f"[INFO] Bỏ qua Pha 2 vì Decoder đã train đủ {args.decoder_epochs} epochs.", flush=True)

    # 3. Đo lường các chỉ số Security: PSNR, SSIM, LPIPS trên tập Test
    print(f"\n--- PHA 3: Đánh giá Chỉ số Security trên Test Set (σ={sigma}) ---", flush=True)
    mean_mse, mean_psnr, mean_ssim, mean_lpips = evaluate_attack(
        client, decoder, defense, testloader, device, MEAN, STD, criterion=criterion_recon, lpips_fn=lpips_fn
    )

    lpips_str = f" | LPIPS: {mean_lpips:.4f}" if mean_lpips is not None else ""
    print(f"[KẾT QUẢ σ={sigma}] Test Acc: {final_test_acc*100:.2f}% | PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}{lpips_str}", flush=True)

    # Lấy một batch ảnh tái tạo mẫu để vẽ lưới so sánh
    sample_originals = None
    sample_recons = None
    with torch.no_grad():
        for x_sample, _ in testloader:
            x_sample = x_sample.to(device)
            sample_originals = x_sample[:8].cpu()
            z_s = client(x_sample[:8])
            z_s_def = defense(z_s)
            sample_recons = decoder(z_s_def).cpu()
            break

    result_summary = {
        "sigma": float(sigma),
        "test_acc": float(final_test_acc),
        "mse": float(mean_mse),
        "psnr": float(mean_psnr),
        "ssim": float(mean_ssim),
        "lpips": float(mean_lpips) if mean_lpips is not None else None,
    }

    # Lưu toàn bộ lịch sử ra file JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_summary": result_summary,
            "history_sl": history_sl,
            "history_dec": history_dec,
        }, f, indent=2)

    return result_summary, sample_originals, sample_recons


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 3: Baseline Phòng thủ B1 (Gaussian Noise) in Split Learning")
    parser.add_argument("--sigma", type=float, default=0.5, help="Mức nhiễu Gaussian sigma đơn lẻ (mặc định: 0.5)")
    parser.add_argument("--sweep", action="store_true", help="Bật chế độ quét đa mức sigma (mặc định: 0.1, 0.5, 1.0)")
    parser.add_argument("--sigmas", type=str, default="0.1,0.5,1.0", help="Danh sách sigmas phân cách bởi dấu phẩy (mặc định: 0.1,0.5,1.0)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện Split Learning (mặc định: 100)")
    parser.add_argument("--decoder-epochs", type=int, default=30, help="Số epochs huấn luyện Decoder tấn công (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate cho Split Learning (mặc định: 0.1)")
    parser.add_argument("--decoder-lr", type=float, default=1e-3, help="Learning rate cho Decoder (mặc định: 1e-3)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá test accuracy (mặc định: 5)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (mặc định: 2)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(STEP0_DIR, "data"), help="Thư mục chứa CIFAR-10")
    parser.add_argument("--output-dir", type=str, default=SCRIPT_DIR, help="Thư mục lưu checkpoints và kết quả")
    parser.add_argument("--no-plot", action="store_true", help="Không tự động xuất đồ thị và lưới ảnh")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint nếu có")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("==================================================", flush=True)
    print("      BƯỚC 3: BASELINE 1 (B1) — GAUSSIAN NOISE DEFENSE", flush=True)
    print("==================================================", flush=True)
    print(f"Thiết bị           : {device}", flush=True)
    if device == "cuda":
        print(f"GPU Name           : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"SL Epochs          : {args.epochs}", flush=True)
    print(f"Decoder Epochs     : {args.decoder_epochs}", flush=True)
    print(f"Batch size         : {args.batch_size}", flush=True)
    print(f"Output Dir         : {args.output_dir}", flush=True)

    if args.sweep:
        target_sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]
        print(f"Chế độ             : Quét đa mức nhiễu (Sweep: {target_sigmas})", flush=True)
    else:
        target_sigmas = [args.sigma]
        print(f"Chế độ             : Chạy mức đơn lẻ (Sigma = {args.sigma})", flush=True)
    print("==================================================\n", flush=True)

    torch.manual_seed(0)
    trainloader, testloader = get_cifar10(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Khởi tạo LPIPS nếu có sẵn
    lpips_fn = get_lpips_fn(device=device)

    all_results = []
    recons_dict = {}
    sample_orig = None

    for sigma in target_sigmas:
        res_summary, origs, recons = run_pipeline_for_sigma(
            sigma, args, trainloader, testloader, device, lpips_fn=lpips_fn
        )
        all_results.append(res_summary)
        recons_dict[sigma] = recons
        if sample_orig is None:
            sample_orig = origs

    # In bảng tổng hợp kết quả (Markdown Table)
    print("\n\n==========================================================================", flush=True)
    print("           BẢNG TỔNG HỢP KẾT QUẢ BASELINE 1: NHIỄU GAUSSIAN (B1)", flush=True)
    print("==========================================================================", flush=True)
    print(f"| {'Sigma (σ)':<10} | {'Test Acc (%)':<14} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'LPIPS':<10} |", flush=True)
    print(f"|{'-'*12}|{'-'*16}|{'-'*14}|{'-'*12}|{'-'*12}|", flush=True)
    for r in all_results:
        lp_str = f"{r['lpips']:.4f}" if r["lpips"] is not None else "N/A"
        print(f"| {r['sigma']:<10.2f} | {r['test_acc']*100:<14.2f} | {r['psnr']:<12.2f} | {r['ssim']:<10.4f} | {lp_str:<10} |", flush=True)
    print("==========================================================================\n", flush=True)

    # Lưu kết quả tổng hợp ra JSON và CSV
    results_json = os.path.join(args.output_dir, "results_b1.json")
    results_csv = os.path.join(args.output_dir, "results_b1.csv")

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sigma", "test_acc", "mse", "psnr", "ssim", "lpips"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"[DONE] Đã lưu bảng kết quả tổng hợp tại: {results_json} & {results_csv}", flush=True)

    # Vẽ đồ thị Trade-off và lưới ảnh tái tạo
    if not args.no_plot:
        plot_tradeoff_path = os.path.join(args.output_dir, "b1_tradeoff_curves.png")
        plot_tradeoff(all_results, save_path=plot_tradeoff_path, show=False)

        if sample_orig is not None and len(recons_dict) > 0:
            plot_recons_path = os.path.join(args.output_dir, "b1_reconstruction_comparison.png")
            plot_reconstruction_grid(sample_orig, recons_dict, MEAN, STD, save_path=plot_recons_path, num_images=6, show=False)

    print("\n>>> HOÀN THÀNH TOÀN BỘ PIPELINE BƯỚC 3 (BASELINE B1)! <<<\n", flush=True)


if __name__ == "__main__":
    main()
