#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py
==============================================================================
HIGH-LEVEL BENCHMARK & REPLICATION ORCHESTRATOR
Điều phối toàn bộ quy trình thực nghiệm và tái lập kết quả khoa học:
1. Huấn luyện Split Learning qua train.py (đa seed, tự động skip nếu đã có checkpoint)
2. Đánh giá tính bảo mật qua eval_attacks.py (Fresh MLP, Fresh Conv, FSHA)
3. Tự động xuất Master Benchmark Table (master_benchmark_table.md & .csv)
==============================================================================
"""
import os
import sys
import time
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
    parser = argparse.ArgumentParser(description="Unified Benchmark & Replication Orchestrator")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "ham10000", "pcam"])
    parser.add_argument("--defense", type=str, default="ar_tape", choices=["none", "b7_lightsplit", "b8_adversarial", "ar_tape"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024], help="Danh sách random seeds")
    parser.add_argument("--k", type=int, default=2048, help="Số chiều bottleneck (mặc định: 2048)")
    parser.add_argument("--lambda-val", type=float, default=0.25, help="Hệ số đối kháng lambda (mặc định: 0.25)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs Split Learning")
    parser.add_argument("--fresh-epochs", type=int, default=30, help="Số epochs đánh giá Fresh Attacker")
    parser.add_argument("--run-fsha", action="store_true", default=True, help="Đánh giá FSHA active attack")
    parser.add_argument("--data-dir", type=str, default="./data", help="Thư mục dữ liệu")
    parser.add_argument("--output-base", type=str, default="./output", help="Thư mục gốc lưu kết quả")
    parser.add_argument("--force", action="store_true", help="Bắt buộc chạy lại từ đầu")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm nhanh")
    args = parser.parse_args()

    python_bin = sys.executable

    print("\n" + "#" * 80)
    print("BẮT ĐẦU QUY TRÌNH BENCHMARK TÁI LẬP (REPLICATION ORCHESTRATOR)")
    print(f"Dataset: {args.dataset.upper()} | Defense: {args.defense.upper()} | Seeds: {args.seeds}")
    print(f"k={args.k} | Lambda={args.lambda_val} | Epochs={args.epochs}")
    print("#" * 80 + "\n", flush=True)

    for seed in args.seeds:
        run_name = f"{args.dataset}_{args.defense}_k{args.k}_lam{args.lambda_val}_s{seed}"
        run_dir = os.path.join(args.output_base, run_name)
        ckpt_path = os.path.join(run_dir, "checkpoints", f"{run_name}_client.pt")
        res_json = os.path.join(run_dir, "results_training.json")

        # 1. Huấn luyện Split Learning
        if os.path.isfile(ckpt_path) and os.path.isfile(res_json) and not args.force:
            print(f"[SKIP / RESUME] Đã tìm thấy checkpoint SL tại {ckpt_path}. Bỏ qua huấn luyện!\n")
        else:
            cmd_train = [
                python_bin, "train.py",
                "--dataset", args.dataset,
                "--defense", args.defense,
                "--k", str(args.k),
                "--lambda-val", str(args.lambda_val),
                "--epochs", str(args.epochs),
                "--seed", str(seed),
                "--data-dir", args.data_dir,
                "--output-dir", run_dir
            ]
            if args.dry_run:
                cmd_train.append("--dry-run")
            run_cmd(cmd_train, desc=f"Huấn luyện Split Learning Seed {seed}")

        # 2. Đánh giá Fresh Attackers
        eval_dir = os.path.join(run_dir, "eval_attacks")
        eval_json = os.path.join(eval_dir, "results_evaluation_attacks.json")
        if os.path.isfile(eval_json) and not args.force:
            print(f"[SKIP / RESUME] Đã tìm thấy kết quả đánh giá tại {eval_json}. Bỏ qua Fresh Evaluation!\n")
        else:
            cmd_eval = [
                python_bin, "eval_attacks.py",
                "--client-ckpt", ckpt_path,
                "--dataset", args.dataset,
                "--k", str(args.k),
                "--epochs", str(args.fresh_epochs),
                "--seed", str(seed),
                "--data-dir", args.data_dir,
                "--output-dir", eval_dir
            ]
            if args.dry_run:
                cmd_eval.append("--dry-run")
            run_cmd(cmd_eval, desc=f"Đánh giá Fresh Attackers Seed {seed}")

    # 3. Tự động cập nhật Bảng Master Table
    print("\n>>> Đang cập nhật Bảng Master Benchmark Table...")
    subprocess.run([python_bin, "scripts/generate_master_table.py"])
    print("\n" + "=" * 80)
    print("HOÀN TẤT TOÀN BỘ QUY TRÌNH BENCHMARK!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
