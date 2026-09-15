# Bước 1 — Tấn công Tái tạo Bị động (Passive Reconstruction Attack) trên Vanilla Split Learning
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

for path in [SCRIPT_DIR, PROJECT_ROOT, STEP0_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.nn as nn
from model import ClientModel
from data import get_cifar10
from decoder import Decoder
from attack import train_decoder_epoch, evaluate_attack
from metrics import get_lpips_fn
from plot_attack import save_reconstruction_grid, plot_attack_curves

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def find_default_client_ckpt():
    candidates = [
        os.path.join(STEP0_DIR, "b0_vanilla.pt"),
        os.path.join(STEP0_DIR, "best_b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "b0_vanilla.pt"),
        os.path.join(SCRIPT_DIR, "b0_vanilla.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 1: Passive Reconstruction Attack on Split Learning IR")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs huấn luyện decoder (mặc định: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate cho Adam (mặc định: 1e-3)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá PSNR/SSIM (mặc định: 5 epochs)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (mặc định: 2)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(STEP0_DIR, "data"), help="Thư mục chứa dữ liệu CIFAR-10")
    parser.add_argument("--client-ckpt", type=str, default=find_default_client_ckpt(), help="Đường dẫn checkpoint Client F_c từ Bước 0")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "b1_decoder.pt"), help="Đường dẫn lưu trọng số Decoder cuối cùng")
    parser.add_argument("--save-best", type=str, default=os.path.join(SCRIPT_DIR, "best_b1_decoder.pt"), help="Đường dẫn lưu Decoder có PSNR cao nhất")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(SCRIPT_DIR, "last_checkpoint_b1.pt"), help="Checkpoint ngắt quãng hỗ trợ resume")
    parser.add_argument("--history-file", type=str, default=os.path.join(SCRIPT_DIR, "step1_history.json"), help="File lưu lịch sử huấn luyện JSON")
    parser.add_argument("--plot-file", type=str, default=os.path.join(SCRIPT_DIR, "attack_curves.png"), help="File lưu đồ thị tấn công")
    parser.add_argument("--grid-file", type=str, default=os.path.join(SCRIPT_DIR, "reconstruction_grid.png"), help="File lưu lưới ảnh đối chứng")
    parser.add_argument("--no-plot", action="store_true", help="Không tự động xuất đồ thị và ảnh lưới")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint gần nhất")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==================================================", flush=True)
    print("      BƯỚC 1: TẤN CÔNG TÁI TẠO BỊ ĐỘNG (IR RECONSTRUCTION)", flush=True)
    print("==================================================", flush=True)
    print(f"Thiết bị         : {device}", flush=True)
    if device == "cuda":
        print(f"GPU Name         : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Số Epochs        : {args.epochs}", flush=True)
    print(f"Batch size       : {args.batch_size}", flush=True)
    print(f"Learning rate    : {args.lr}", flush=True)
    print(f"Eval Frequency   : every {args.eval_freq} epochs", flush=True)
    print(f"Client Checkpoint: {args.client-ckpt if hasattr(args, 'client-ckpt') else args.client_ckpt}", flush=True)
    print(f"Output Decoder   : {args.output}", flush=True)
    print(f"Best Decoder     : {args.save_best}", flush=True)
    print("==================================================\n", flush=True)

    torch.manual_seed(0)
    trainloader, testloader = get_cifar10(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # 1. Khởi tạo và nạp trọng số Client F_c (BƯỚC 0)
    client = ClientModel().to(device)
    if os.path.isfile(args.client_ckpt):
        print(f"[INFO] Nạp trọng số Client F_c từ: {args.client_ckpt}", flush=True)
        ckpt = torch.load(args.client_ckpt, map_location=device)
        if "client" in ckpt:
            client.load_state_dict(ckpt["client"])
        else:
            client.load_state_dict(ckpt)
    else:
        print(f"[WARN] Không tìm thấy checkpoint Bước 0 tại: {args.client_ckpt}", flush=True)
        print("       Đang dùng khởi tạo ngẫu nhiên của ClientModel. (Khuyến nghị: Chạy train Bước 0 trước)", flush=True)

    # Đóng băng vĩnh viễn F_c (tấn công bị động)
    client.eval()
    for param in client.parameters():
        param.requires_grad = False

    # 2. Khởi tạo Decoder D_phi và Optimizer Adam
    decoder = Decoder().to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Thử khởi tạo hàm tính khoảng cách cảm nhận LPIPS
    lpips_fn = get_lpips_fn(device=device)
    if lpips_fn is not None:
        print("[INFO] Đã nạp thành công LPIPS (AlexNet) để đánh giá độ tương đồng cảm nhận.", flush=True)
    else:
        print("[INFO] LPIPS chưa sẵn sàng (có thể cài đặt qua `pip install lpips`). Sẽ đo PSNR & SSIM.", flush=True)

    start_epoch = 1
    best_psnr = 0.0
    best_ssim = 0.0
    history = []

    # Khôi phục nếu có cờ --resume
    if args.resume and os.path.isfile(args.checkpoint):
        print(f"[RESUME] Nạp checkpoint từ: {args.checkpoint}", flush=True)
        ckpt_resume = torch.load(args.checkpoint, map_location=device)
        decoder.load_state_dict(ckpt_resume["decoder"])
        opt.load_state_dict(ckpt_resume["opt"])
        start_epoch = ckpt_resume["epoch"] + 1
        best_psnr = ckpt_resume.get("best_psnr", 0.0)
        best_ssim = ckpt_resume.get("best_ssim", 0.0)
        history = ckpt_resume.get("history", [])
        if not history and os.path.isfile(args.history_file):
            try:
                with open(args.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        print(f"[RESUME] Tiếp tục từ epoch {start_epoch} (Best PSNR: {best_psnr:.2f} dB, Best SSIM: {best_ssim:.4f})", flush=True)

    print("\nBắt đầu huấn luyện Decoder tấn công tái tạo...\n", flush=True)
    total_start = time.time()
    csv_file = os.path.splitext(args.history_file)[0] + ".csv"

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_mse = train_decoder_epoch(client, decoder, trainloader, opt, criterion, device)
        epoch_time = time.time() - t0

        test_mse = None
        test_psnr = None
        test_ssim = None
        test_lpips = None
        eval_time = 0.0
        best_tag = ""

        is_eval_epoch = (epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs)
        if is_eval_epoch:
            t_eval_0 = time.time()
            test_mse, test_psnr, test_ssim, test_lpips = evaluate_attack(
                client, decoder, testloader, device, MEAN, STD, criterion=criterion, lpips_fn=lpips_fn
            )
            eval_time = time.time() - t_eval_0

            is_best = test_psnr > best_psnr
            if is_best:
                best_psnr = test_psnr
                best_ssim = test_ssim
                os.makedirs(os.path.dirname(os.path.abspath(args.save_best)), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "decoder": decoder.state_dict(),
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "opt": opt.state_dict(),
                    "history": history + [{
                        "epoch": epoch,
                        "train_mse": float(train_mse),
                        "test_mse": float(test_mse),
                        "test_psnr": float(test_psnr),
                        "test_ssim": float(test_ssim),
                        "test_lpips": float(test_lpips) if test_lpips is not None else None,
                        "epoch_time": float(epoch_time + eval_time),
                    }],
                }, args.save_best)
                best_tag = " -> [BEST SAVED]"

            lpips_str = f" | LPIPS: {test_lpips:.4f}" if test_lpips is not None else ""
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train MSE: {train_mse:.5f} | Test MSE: {test_mse:.5f} | PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f}{lpips_str} | Train: {epoch_time:.1f}s | Eval: {eval_time:.1f}s{best_tag}", flush=True)
        else:
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train MSE: {train_mse:.5f} | Train: {epoch_time:.1f}s", flush=True)

        epoch_entry = {
            "epoch": epoch,
            "train_mse": float(train_mse),
            "test_mse": float(test_mse) if test_mse is not None else None,
            "test_psnr": float(test_psnr) if test_psnr is not None else None,
            "test_ssim": float(test_ssim) if test_ssim is not None else None,
            "test_lpips": float(test_lpips) if test_lpips is not None else None,
            "epoch_time": float(epoch_time + eval_time),
        }
        history.append(epoch_entry)

        # Lưu lịch sử JSON
        os.makedirs(os.path.dirname(os.path.abspath(args.history_file)), exist_ok=True)
        with open(args.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Lưu lịch sử CSV
        try:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "test_mse", "test_psnr", "test_ssim", "test_lpips", "epoch_time"])
                writer.writeheader()
                writer.writerows(history)
        except Exception:
            pass

        # Lưu checkpoint ngắt quãng sau mỗi epoch
        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "decoder": decoder.state_dict(),
            "opt": opt.state_dict(),
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "history": history,
        }, args.checkpoint)

        # Cập nhật đồ thị và ảnh lưới định kỳ
        if not args.no_plot and (is_eval_epoch or epoch == args.epochs):
            try:
                plot_attack_curves(history, save_path=args.plot_file, show=False)
                save_reconstruction_grid(client, decoder, testloader, device, MEAN, STD, save_path=args.grid_file, num_images=8)
            except Exception as e:
                print(f"[WARN] Lỗi khi cập nhật ảnh đồ thị: {e}", flush=True)

    total_time = time.time() - total_start
    print(f"\n==================================================", flush=True)
    print(f"Huấn luyện Decoder hoàn thành trong {total_time/60:.2f} phút.", flush=True)
    print(f"🌟 Best PSNR đạt được : {best_psnr:.2f} dB", flush=True)
    print(f"🌟 Best SSIM đạt được : {best_ssim:.4f}", flush=True)

    # Đánh giá Acceptance Criteria
    psnr_pass = best_psnr > 20.0
    ssim_pass = best_ssim > 0.60
    print("\n--- KIỂM TRA TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA) ---", flush=True)
    print(f"1. PSNR > 20 dB : {'✅ ĐẠT' if psnr_pass else '❌ CHƯA ĐẠT'} ({best_psnr:.2f} dB)", flush=True)
    print(f"2. SSIM > 0.60  : {'✅ ĐẠT' if ssim_pass else '❌ CHƯA ĐẠT'} ({best_ssim:.4f})", flush=True)
    if psnr_pass and ssim_pass:
        print(">>> KẾT LUẬN: LỖ HỔNG BỊ ĐẢO NGƯỢC CỦA VANILLA SL ĐƯỢC XÁC NHẬN HOÀN TOÀN! <<<", flush=True)
    print("==================================================\n", flush=True)

    # Xuất lần cuối cùng
    if not args.no_plot:
        try:
            plot_attack_curves(history, save_path=args.plot_file, show=False)
            save_reconstruction_grid(client, decoder, testloader, device, MEAN, STD, save_path=args.grid_file, num_images=8)
        except Exception:
            pass

    # Lưu trọng số tham chiếu cuối cùng
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(decoder.state_dict(), args.output)
    print(f"[DONE] Đã lưu trọng số decoder tại: {args.output}", flush=True)
    print(f"[DONE] Đã lưu lịch sử tại: {args.history_file} & {csv_file}", flush=True)
    if not args.no_plot:
        print(f"[DONE] Đã lưu đồ thị tại: {args.plot_file}", flush=True)
        print(f"[DONE] Đã lưu ảnh lưới đối chứng tại: {args.grid_file}\n", flush=True)


if __name__ == "__main__":
    main()
