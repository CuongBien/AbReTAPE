#!/usr/bin/env python3
# ==============================================================================
# RUNNER PYTHON CHO AR-TAPE v1 UTILITY SWEEP (LAMBDA = 0.1 & 0.25)
# ==============================================================================
# - Mục tiêu: Tìm điểm vận hành Pareto tối ưu (Acc ~91% với SSIM < 0.30)
# - Cấu hình: k=2048, Mode F (fixed_bottleneck, linear_fixed), CosineAnnealingLR,
#             warmup 10 epochs, seed 42, 100 epochs Split Learning
# - Đánh giá: Fresh MLP (30 epochs) & Fresh Conv (30 epochs) tại epoch cuối (không chọn theo test)
# ==============================================================================
import os
import sys
import time
import json
import argparse
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd_list, desc=""):
    print("\n" + "=" * 80)
    print(f">>> {desc}")
    print(f"Lệnh: {' '.join(cmd_list)}")
    print("=" * 80 + "\n", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_list)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Lỗi khi thực thi: {desc} (Exit code: {res.returncode})", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"[HOÀN TẤT] {desc} trong {dt:.1f}s\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Quét Utility Sweep AR-TAPE v1 cho lambda 0.1 và 0.25")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.25], help="Danh sách lambda (mặc định: 0.1 0.25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (mặc định: 42)")
    parser.add_argument("--k", type=int, default=2048, help="Bottleneck dimension (mặc định: 2048)")
    parser.add_argument("--m", type=int, default=512, help="Unprojector hidden dimension (mặc định: 512)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Thư mục CIFAR-10")
    parser.add_argument("--base-drive-dir", type=str, default="/content/drive/MyDrive/AbReTAPE_Results",
                        help="Thư mục gốc lưu kết quả trên Google Drive")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs Split Learning")
    parser.add_argument("--fresh-epochs", type=int, default=30, help="Số epochs đánh giá Fresh Attacker")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm rút gọn (1 epoch)")
    args = parser.parse_args()

    python_bin = sys.executable
    sweep_root = os.path.join(args.base_drive_dir, "AR_TAPE_v1_Utility_Sweep")
    os.makedirs(sweep_root, exist_ok=True)

    summary_records = []

    for lam in args.lambdas:
        print(f"\n################################################################################")
        print(f"BẮT ĐẦU QUÁ TRÌNH CHO LAMBDA = {lam} (SEED {args.seed})")
        print(f"################################################################################\n", flush=True)

        sl_out = os.path.join(sweep_root, f"lam{lam}_s{args.seed}")
        fresh_out = os.path.join(sweep_root, f"fresh_lam{lam}_s{args.seed}")
        os.makedirs(sl_out, exist_ok=True)
        os.makedirs(fresh_out, exist_ok=True)

        ckpt_path = os.path.join(sl_out, "checkpoints", f"artape_v0_k{args.k}_linear_fixed_fixR_lam{lam}_s{args.seed}_client.pt")

        # ----------------------------------------------------------------------
        # 1. Huấn luyện Split Learning (100 epochs, Cosine Scheduler)
        # ----------------------------------------------------------------------
        cmd_sl = [
            python_bin, "train_artape_v0.py",
            "--k", str(args.k),
            "--m", str(args.m),
            "--unprojector-mode", "linear_fixed",
            "--fixed-bottleneck",
            "--lambdas", str(lam),
            "--warmup-epochs", "10",
            "--scheduler", "cosine",
            "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--batch-size", "128",
            "--lr", "0.1",
            "--adv-lr", "1e-3",
            "--cap", "1.5586",
            "--data-dir", args.data_dir,
            "--output-dir", sl_out
        ]
        if args.dry_run:
            cmd_sl.append("--dry-run")

        run_cmd(cmd_sl, desc=f"[1/2] Huấn luyện Split Learning Cosine λ={lam} (Seed {args.seed})")

        # ----------------------------------------------------------------------
        # 2. Đánh giá Fresh Attackers (MLP và Conv, 30 epochs)
        # ----------------------------------------------------------------------
        cmd_fresh = [
            python_bin, "eval_artape_v0_fresh.py",
            "--k", str(args.k),
            "--fixed-bottleneck",
            "--client-ckpt", ckpt_path,
            "--attackers", "mlp", "conv",
            "--lambdas", str(lam),
            "--seed", str(args.seed),
            "--epochs", str(args.fresh_epochs),
            "--batch-size", "128",
            "--lr", "1e-3",
            "--data-dir", args.data_dir,
            "--output-dir", fresh_out
        ]
        if args.dry_run:
            cmd_fresh.append("--dry-run")

        run_cmd(cmd_fresh, desc=f"[2/2] Đánh giá Fresh MLP & Conv λ={lam} (Seed {args.seed})")

        # ----------------------------------------------------------------------
        # 3. Thu thập kết quả trực tiếp từ JSON
        # ----------------------------------------------------------------------
        sl_json = os.path.join(sl_out, "results_artape_v0.json")
        fresh_json = os.path.join(fresh_out, "results_artape_v0_fresh_attackers.json")

        acc_10k, acc_2k = None, None
        if os.path.isfile(sl_json):
            with open(sl_json, "r", encoding="utf-8") as f:
                sl_d = json.load(f)
            key = list(sl_d.keys())[0]
            acc_10k = sl_d[key].get("final_full_test_acc")
            acc_2k = sl_d[key].get("mean_task_acc_last10_2k")

        mlp_ssim_10k, conv_ssim_10k = None, None
        mlp_psnr_10k, conv_psnr_10k = None, None
        if os.path.isfile(fresh_json):
            with open(fresh_json, "r", encoding="utf-8") as f:
                f_d = json.load(f)
            for item in f_d:
                if item["attacker_type"] == "mlp":
                    mlp_ssim_10k = item.get("fresh_ssim_10k")
                    mlp_psnr_10k = item.get("fresh_psnr_10k")
                elif item["attacker_type"] == "conv":
                    conv_ssim_10k = item.get("fresh_ssim_10k")
                    conv_psnr_10k = item.get("fresh_psnr_10k")

        rec = {
            "lambda": lam,
            "seed": args.seed,
            "acc_10k": acc_10k,
            "acc_2k": acc_2k,
            "mlp_ssim_10k": mlp_ssim_10k,
            "mlp_psnr_10k": mlp_psnr_10k,
            "conv_ssim_10k": conv_ssim_10k,
            "conv_psnr_10k": conv_psnr_10k
        }
        summary_records.append(rec)

    # --------------------------------------------------------------------------
    # 4. Xuất Báo Cáo Tổng Hợp & In Bảng Trade-off
    # --------------------------------------------------------------------------
    summary_json_path = os.path.join(sweep_root, "summary_utility_sweep.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"BẢNG ĐỐI CHUẨN UTILITY SWEEP AR-TAPE v1 (Cosine, k=2048, Seed {args.seed}):")
    print("=" * 90)
    print(f"{'λ':<6} | {'Acc (10k Full)':<15} | {'Acc (2k Fast)':<15} | {'Fresh MLP SSIM':<16} | {'Fresh Conv SSIM':<16} | {'Nhận Định'}")
    print("-" * 90)
    for r in summary_records:
        acc10 = f"{r['acc_10k']:.2f}%" if r['acc_10k'] is not None else "N/A"
        acc2 = f"{r['acc_2k']:.2f}%" if r['acc_2k'] is not None else "N/A"
        mlp_s = f"{r['mlp_ssim_10k']:.4f}" if r['mlp_ssim_10k'] is not None else "N/A"
        conv_s = f"{r['conv_ssim_10k']:.4f}" if r['conv_ssim_10k'] is not None else "N/A"

        verdict = []
        if r['acc_10k'] and r['acc_10k'] >= 90.5:
            verdict.append("Acc ≥ 90.5%")
        if r['mlp_ssim_10k'] and r['mlp_ssim_10k'] < 0.30 and r['conv_ssim_10k'] and r['conv_ssim_10k'] < 0.30:
            verdict.append("SSIM < 0.30")
        v_str = " & ".join(verdict) if verdict else "Điểm trên đường đánh đổi"

        print(f"{r['lambda']:<6} | {acc10:<15} | {acc2:<15} | {mlp_s:<16} | {conv_s:<16} | {v_str}")
    print("=" * 90)
    print(f"Đã lưu kết quả tại: {summary_json_path}\n")


if __name__ == "__main__":
    main()
