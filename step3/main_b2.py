# Bước 3 — Baseline 2 (B2): Phòng thủ DP-SGD (Differential Privacy) trên Client và Tấn công Tái tạo Thích ứng
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
from dpsgd import DPSGDClientOptimizer, compute_dp_epsilon
from train_b2 import train_sl_dp_epoch, evaluate_sl_b2
from attack_b2 import train_decoder_b2_epoch, evaluate_attack_b2
from plot_b2 import plot_b2_tradeoff, plot_b2_reconstruction_grid

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def run_pipeline_for_sigma_dp(sigma_dp, args, trainloader, testloader, device, lpips_fn=None):
    """
    Chạy toàn bộ pipeline thống nhất cho 1 mức sigma_DP:
    1. Huấn luyện Split Learning với Client dùng DP-SGD (cắt ngưỡng C, cộng nhiễu sigma_DP).
    2. Đóng băng Client.
    3. Huấn luyện Decoder tấn công tái tạo thích ứng trên IR z = F_c(x).
    4. Đo lường Test Classification Accuracy (Utility) và PSNR/SSIM/LPIPS (Security).
    """
    print(f"\n==================================================", flush=True)
    print(f"   BƯỚC 3 (B2): HUẤN LUYỆN DP-SGD VỚI SIGMA = {sigma_dp}", flush=True)
    print(f"==================================================", flush=True)

    # Tính ngân sách bảo mật DP theo công thức Rényi Differential Privacy
    dataset_size = len(trainloader.dataset)
    dp_epsilon = compute_dp_epsilon(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_size=dataset_size,
        noise_multiplier=sigma_dp,
        delta=args.delta
    )
    print(f"[PRIVACY BUDGET] Ngân sách bảo mật RDP: (ε = {dp_epsilon:.2f}, δ = {args.delta:.1e}) sau {args.epochs} epochs", flush=True)

    ckpt_path = os.path.join(args.output_dir, f"b2_sigma_{sigma_dp}.pt")
    dec_ckpt_path = os.path.join(args.output_dir, f"decoder_b2_sigma_{sigma_dp}.pt")
    history_path = os.path.join(args.output_dir, f"history_b2_sigma_{sigma_dp}.json")

    # 1. Khởi tạo mô hình
    client = ClientModel().to(device)
    server = ServerModel().to(device)

    opt_c_raw = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    micro_size = args.micro_batch_size if args.micro_batch_size > 0 else None
    dp_client = DPSGDClientOptimizer(
        client=client,
        optimizer=opt_c_raw,
        max_grad_norm=args.clip_norm,
        noise_multiplier=sigma_dp,
        micro_batch_size=micro_size
    )

    sched_c = torch.optim.lr_scheduler.MultiStepLR(opt_c_raw, milestones=[50, 75], gamma=0.1)
    sched_s = torch.optim.lr_scheduler.MultiStepLR(opt_s, milestones=[50, 75], gamma=0.1)
    criterion_task = nn.CrossEntropyLoss()

    start_epoch = 1
    best_acc = 0.0
    history_sl = []

    # Kiểm tra checkpoint resume nếu có
    if args.resume and os.path.isfile(ckpt_path):
        print(f"[RESUME] Nạp checkpoint SL DP-SGD từ: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        client.load_state_dict(ckpt["client"])
        server.load_state_dict(ckpt["server"])
        if "opt_c" in ckpt:
            opt_c_raw.load_state_dict(ckpt["opt_c"])
        if "opt_s" in ckpt:
            opt_s.load_state_dict(ckpt["opt_s"])
        if "sched_c" in ckpt:
            sched_c.load_state_dict(ckpt["sched_c"])
        if "sched_s" in ckpt:
            sched_s.load_state_dict(ckpt["sched_s"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        history_sl = ckpt.get("history_sl", [])

    # Huấn luyện Split Learning với Client DP-SGD
    if start_epoch <= args.epochs:
        mode_str = f"micro-batch={micro_size}" if micro_size else "batch-level"
        print(f"\n--- PHA 1: Huấn luyện Split Learning với Client DP-SGD (σ_DP={sigma_dp}, C={args.clip_norm}, {mode_str}) ---", flush=True)
        t_sl_start = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            loss_train, acc_train = train_sl_dp_epoch(client, server, dp_client, trainloader, opt_s, criterion_task, device)
            curr_lr = opt_c_raw.param_groups[0]["lr"]
            sched_c.step()
            sched_s.step()
            epoch_time = time.time() - t0

            is_eval = (epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs)
            loss_val, acc_val = (None, None)
            if is_eval:
                loss_val, acc_val = evaluate_sl_b2(client, server, testloader, device, criterion=criterion_task)
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

            torch.save({
                "epoch": epoch,
                "sigma_dp": sigma_dp,
                "clip_norm": args.clip_norm,
                "epsilon": dp_epsilon,
                "delta": args.delta,
                "client": client.state_dict(),
                "server": server.state_dict(),
                "opt_c": opt_c_raw.state_dict(),
                "opt_s": opt_s.state_dict(),
                "sched_c": sched_c.state_dict(),
                "sched_s": sched_s.state_dict(),
                "best_acc": best_acc,
                "history_sl": history_sl,
            }, ckpt_path)

        print(f"[DONE] Huấn luyện SL DP-SGD (σ_DP={sigma_dp}) hoàn thành trong {(time.time() - t_sl_start)/60:.2f} phút. Best Acc: {best_acc*100:.2f}%", flush=True)
    else:
        print(f"[INFO] Bỏ qua Pha 1 vì đã train đủ {args.epochs} epochs.", flush=True)

    # Đo độ chính xác cuối cùng trên Test set
    _, final_test_acc = evaluate_sl_b2(client, server, testloader, device, criterion=criterion_task)

    # 2. Huấn luyện Decoder tấn công tái tạo thích ứng trên Client đã đóng băng
    print(f"\n--- PHA 2: Tấn công Tái tạo Thích ứng trên Client DP-SGD (σ_DP={sigma_dp}) ---", flush=True)
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
            loss_dec = train_decoder_b2_epoch(client, decoder, trainloader, opt_d, criterion_recon, device)
            d_time = time.time() - t0

            is_eval_dec = (d_epoch % 5 == 0 or d_epoch == 1 or d_epoch == args.decoder_epochs)
            if is_eval_dec:
                print(f"Decoder Epoch {d_epoch:2d}/{args.decoder_epochs} | Recon Loss (MSE): {loss_dec:.5f} | Time: {d_time:.1f}s", flush=True)
            history_dec.append({"epoch": d_epoch, "loss": float(loss_dec)})

            torch.save({
                "epoch": d_epoch,
                "sigma_dp": sigma_dp,
                "decoder": decoder.state_dict(),
                "opt_d": opt_d.state_dict(),
                "history_dec": history_dec,
            }, dec_ckpt_path)

        print(f"[DONE] Huấn luyện Decoder (σ_DP={sigma_dp}) hoàn thành trong {(time.time() - t_dec_start)/60:.2f} phút.", flush=True)
    else:
        print(f"[INFO] Bỏ qua Pha 2 vì Decoder đã train đủ {args.decoder_epochs} epochs.", flush=True)

    # 3. Đo lường Security (PSNR, SSIM, LPIPS)
    print(f"\n--- PHA 3: Đánh giá Chỉ số Security trên Test Set (σ_DP={sigma_dp}) ---", flush=True)
    mean_mse, mean_psnr, mean_ssim, mean_lpips = evaluate_attack_b2(
        client, decoder, testloader, device, MEAN, STD, criterion=criterion_recon, lpips_fn=lpips_fn
    )

    lpips_str = f" | LPIPS: {mean_lpips:.4f}" if mean_lpips is not None else ""
    print(f"[KẾT QUẢ σ_DP={sigma_dp}] Test Acc: {final_test_acc*100:.2f}% | PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}{lpips_str}", flush=True)

    # Lấy mẫu ảnh phục vụ vẽ lưới so sánh
    sample_originals = None
    sample_recons = None
    with torch.no_grad():
        for x_sample, _ in testloader:
            x_sample = x_sample.to(device)
            sample_originals = x_sample[:8].cpu()
            z_s = client(x_sample[:8])
            sample_recons = decoder(z_s).cpu()
            break

    result_summary = {
        "sigma_dp": float(sigma_dp),
        "clip_norm": float(args.clip_norm),
        "epsilon": float(dp_epsilon),
        "delta": float(args.delta),
        "test_acc": float(final_test_acc),
        "mse": float(mean_mse),
        "psnr": float(mean_psnr),
        "ssim": float(mean_ssim),
        "lpips": float(mean_lpips) if mean_lpips is not None else None,
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_summary": result_summary,
            "history_sl": history_sl,
            "history_dec": history_dec,
        }, f, indent=2)

    return result_summary, sample_originals, sample_recons


def main():
    parser = argparse.ArgumentParser(description="Bước 3 — Baseline 2 (B2): Client DP-SGD Defense & Adaptive Decoder Attack")
    parser.add_argument("--epochs", type=int, default=100, help="Số epoch huấn luyện Split Learning (mặc định: 100)")
    parser.add_argument("--decoder-epochs", type=int, default=30, help="Số epoch huấn luyện Decoder (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.05, help="Tốc độ học cho SL (mặc định: 0.05)")
    parser.add_argument("--decoder-lr", type=float, default=1e-3, help="Tốc độ học cho Decoder (mặc định: 0.001)")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Ngưỡng cắt chuẩn gradient C cho DP-SGD (mặc định: 1.0)")
    parser.add_argument("--sigma-dp", type=float, default=1.0, help="Hệ số nhiễu Gauss sigma_DP khi chạy đơn lẻ (mặc định: 1.0)")
    parser.add_argument("--micro-batch-size", type=int, default=16, help="Kích thước micro-batch để cắt gradient (mặc định: 16; đặt <=0 để cắt toàn bộ batch)")
    parser.add_argument("--delta", type=float, default=1e-5, help="Ngưỡng xác suất vi phạm DP delta (mặc định: 1e-5)")
    parser.add_argument("--sweep", action="store_true", help="Chạy quét toàn diện các mức sigma_DP (0.5, 1.0, 2.0)")
    parser.add_argument("--sigmas-dp", type=str, default="0.5,1.0,2.0", help="Danh sách sigma_DP quét, phân tách bởi dấu phẩy")
    parser.add_argument("--data-dir", type=str, default="./data", help="Đường dẫn thư mục chứa dataset CIFAR-10")
    parser.add_argument("--output-dir", type=str, default="checkpoints_b2", help="Thư mục lưu checkpoints và kết quả B2")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá validation (epoch)")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint gần nhất nếu có")
    parser.add_argument("--no-lpips", action="store_true", help="Bỏ qua tính toán metric LPIPS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Thiết lập seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị thực thi: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Nạp dữ liệu CIFAR-10
    print(f"[INFO] Đang nạp dataset CIFAR-10 từ: {args.data_dir} (Batch size: {args.batch_size})")
    trainloader, testloader = get_cifar10(data_dir=args.data_dir, batch_size=args.batch_size)

    # Khởi tạo LPIPS nếu khả dụng
    lpips_fn = None
    if not args.no_lpips:
        try:
            lpips_fn = get_lpips_fn(device)
            print("[INFO] Đã nạp thành công mạng LPIPS (VGG).")
        except Exception as e:
            print(f"[WARN] Không thể nạp LPIPS ({e}). Sẽ bỏ qua đo lường LPIPS.")

    # Xác định danh sách sigma_DP cần chạy
    if args.sweep:
        sigmas = [float(s.strip()) for s in args.sigmas_dp.split(",") if s.strip()]
        print(f"\n>>> CHẾ ĐỘ QUÉT ĐA MỨC NHIỄU DP-SGD (SWEEP): {sigmas} <<<", flush=True)
    else:
        sigmas = [args.sigma_dp]
        print(f"\n>>> CHẾ ĐỘ CHẠY ĐƠN LẺ: sigma_DP = {args.sigma_dp} <<<", flush=True)

    all_results = []
    recons_by_sigma = {}
    eps_by_sigma = {}
    cached_originals = None

    for s_dp in sigmas:
        res, originals, recons = run_pipeline_for_sigma_dp(s_dp, args, trainloader, testloader, device, lpips_fn=lpips_fn)
        all_results.append(res)
        recons_by_sigma[s_dp] = recons
        eps_by_sigma[s_dp] = res["epsilon"]
        if cached_originals is None and originals is not None:
            cached_originals = originals

    # In bảng tổng kết định dạng Markdown
    print(f"\n{'='*75}")
    print("   BẢNG TỔNG KẾT KẾT QUẢ ĐÁNH GIÁ BASELINE B2 (DP-SGD ON CLIENT)")
    print(f"{'='*75}")
    header = "| sigma_DP | Epsilon (ε) | Clip Norm C | Test Acc (%) | PSNR (dB) | SSIM |"
    sep    = "|:--------:|:-----------:|:-----------:|:------------:|:---------:|:----:|"
    if not args.no_lpips and any(r["lpips"] is not None for r in all_results):
        header += " LPIPS |"
        sep    += "------:|"
    print(header)
    print(sep)

    for r in all_results:
        row = f"| {r['sigma_dp']:8.2f} | {r['epsilon']:11.2f} | {r['clip_norm']:11.1f} | {r['test_acc']*100:12.2f} | {r['psnr']:9.2f} | {r['ssim']:4.4f} |"
        if not args.no_lpips and r["lpips"] is not None:
            row += f" {r['lpips']:5.4f} |"
        print(row)
    print(f"{'='*75}\n")

    # Lưu kết quả ra file JSON và CSV
    results_json_path = os.path.join(args.output_dir, "results_b2.json")
    results_csv_path = os.path.join(args.output_dir, "results_b2.csv")

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[SAVE] Đã lưu kết quả chi tiết tại: {results_json_path}")

    fieldnames = ["sigma_dp", "clip_norm", "epsilon", "delta", "test_acc", "mse", "psnr", "ssim", "lpips"]
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"[SAVE] Đã lưu kết quả bảng số liệu tại: {results_csv_path}")

    # Vẽ đồ thị Trade-off và xuất lưới so sánh ảnh tái tạo
    tradeoff_img_path = os.path.join(args.output_dir, "b2_tradeoff_curves.png")
    grid_img_path = os.path.join(args.output_dir, "b2_reconstruction_comparison.png")

    plot_b2_tradeoff(all_results, save_path=tradeoff_img_path)
    if cached_originals is not None and recons_by_sigma:
        plot_b2_reconstruction_grid(
            cached_originals, 
            recons_by_sigma, 
            eps_by_sigma=eps_by_sigma,
            mean=MEAN, 
            std=STD, 
            save_path=grid_img_path, 
            num_images=6
        )

    print(f"[COMPLETED] Toàn bộ pipeline Baseline B2 đã kết thúc thành công!\n")


if __name__ == "__main__":
    main()
