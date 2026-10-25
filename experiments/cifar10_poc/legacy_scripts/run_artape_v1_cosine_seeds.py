#!/usr/bin/env python3
# ==============================================================================
# RUNNER PYTHON CHO AR-TAPE v1 (COSINE SCHEDULER + FRESH ATTACKERS) TRÊN ĐA SEED
# ==============================================================================
import os
import sys
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
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        print(f"[ERROR] Lỗi khi thực thi: {desc} (Exit code: {res.returncode})", file=sys.stderr)
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="Chạy tự động AR-TAPE v1 Cosine và Fresh Eval trên nhiều seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 2024], help="Danh sách random seed (mặc định: 7 2024)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Thư mục chứa dữ liệu CIFAR-10")
    parser.add_argument("--base-drive-dir", type=str, default="/content/drive/MyDrive/AbReTAPE_Results", 
                        help="Thư mục gốc lưu kết quả trên Google Drive")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs Split Learning")
    parser.add_argument("--fresh-epochs", type=int, default=30, help="Số epochs đánh giá Fresh Attacker")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm rút gọn")
    args = parser.parse_args()

    python_bin = sys.executable

    for seed in args.seeds:
        print(f"\n################################################################################")
        print(f"BẮT ĐẦU QUÁ TRÌNH CHO SEED {seed}")
        print(f"################################################################################\n", flush=True)

        sl_out = os.path.join(args.base_drive_dir, "AR_TAPE_v1_Cosine", f"lam0.5_s{seed}")
        fresh_out = os.path.join(args.base_drive_dir, "AR_TAPE_v1_Cosine_Fresh", f"fresh_lam0.5_s{seed}")
        ckpt_path = os.path.join(sl_out, "checkpoints", f"artape_v0_k2048_linear_fixed_fixR_lam0.5_s{seed}_client.pt")

        # 1. Huấn luyện Split Learning với Cosine Scheduler
        cmd_sl = [
            python_bin, "train_artape_v0.py",
            "--k", "2048",
            "--m", "512",
            "--unprojector-mode", "linear_fixed",
            "--fixed-bottleneck",
            "--lambdas", "0.5",
            "--warmup-epochs", "10",
            "--scheduler", "cosine",
            "--seed", str(seed),
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
        
        run_cmd(cmd_sl, desc=f"[1/2] Huấn luyện Split Learning Cosine λ=0.5 (Seed {seed})")

        # 2. Đánh giá Fresh Attackers (MLP và Conv)
        cmd_fresh = [
            python_bin, "eval_artape_v0_fresh.py",
            "--k", "2048",
            "--fixed-bottleneck",
            "--client-ckpt", ckpt_path,
            "--attackers", "mlp", "conv",
            "--lambdas", "0.5",
            "--seed", str(seed),
            "--epochs", str(args.fresh_epochs),
            "--batch-size", "128",
            "--lr", "1e-3",
            "--data-dir", args.data_dir,
            "--output-dir", fresh_out
        ]
        if args.dry_run:
            cmd_fresh.append("--dry-run")

        run_cmd(cmd_fresh, desc=f"[2/2] Đánh giá Fresh MLP & Conv trên Client tĩnh (Seed {seed})")

    print("\n" + "=" * 80)
    print(f"[HOÀN TẤT] Toàn bộ các seed {args.seeds} đã được hoàn thành thành công!")
    print(f"Kết quả lưu tại: {args.base_drive_dir}")
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    main()
