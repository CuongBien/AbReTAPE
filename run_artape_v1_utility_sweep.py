#!/usr/bin/env python3
# ==============================================================================
# RUNNER PYTHON CHO AR-TAPE v1 UTILITY SWEEP & FSHA (MULTI-SEED & AUTOMATION)
# ==============================================================================
# - Mục tiêu: Đóng PoC toàn diện cho AR-TAPE (ResNet-18 Cut-1, CIFAR-10, Mode F k=2048)
# - Cấu hình: CosineAnnealingLR, warmup 10 epochs, 100 epochs Split Learning
# - Hỗ trợ:
#     * Quét đa seed: --seeds 42 7 2024
#     * Tự động chạy Fresh MLP/Conv (30 epochs, final epoch metrics)
#     * Tự động chạy FSHA (--run-fsha: gs=1.0 & gs=5.0, 30 epochs, final epoch metrics)
#     * Cơ chế Resume / Skip thông minh tránh tính toán trùng lặp
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
    parser = argparse.ArgumentParser(description="Quét Utility Sweep & FSHA AR-TAPE v1")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.25], help="Danh sách lambda (vd: 0.25 hoặc 0.1 0.25)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Danh sách seeds (vd: 7 2024 hoặc 42 7 2024)")
    parser.add_argument("--seed", type=int, default=42, help="Seed đơn lẻ (nếu không truyền --seeds)")
    parser.add_argument("--k", type=int, default=2048, help="Bottleneck dimension (mặc định: 2048)")
    parser.add_argument("--m", type=int, default=512, help="Unprojector hidden dimension (mặc định: 512)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Thư mục CIFAR-10")
    parser.add_argument("--base-drive-dir", type=str, default="/content/drive/MyDrive/AbReTAPE_Results",
                        help="Thư mục gốc lưu kết quả trên Google Drive")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs Split Learning")
    parser.add_argument("--fresh-epochs", type=int, default=30, help="Số epochs đánh giá Fresh Attacker")
    parser.add_argument("--fsha-epochs", type=int, default=30, help="Số epochs đánh giá FSHA")
    parser.add_argument("--run-fsha", action="store_true", help="Chạy đánh giá FSHA chủ động (gs=1.0 & gs=5.0)")
    parser.add_argument("--force", action="store_true", help="Bắt buộc chạy lại từ đầu kể cả khi checkpoint đã tồn tại")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử nghiệm rút gọn (1 epoch)")
    args = parser.parse_args()

    python_bin = sys.executable
    sweep_root = os.path.join(args.base_drive_dir, "AR_TAPE_v1_Utility_Sweep")
    os.makedirs(sweep_root, exist_ok=True)

    seeds_list = args.seeds if args.seeds is not None else [args.seed]
    summary_records = []

    for lam in args.lambdas:
        for seed in seeds_list:
            print(f"\n################################################################################")
            print(f"BẮT ĐẦU QUÁ TRÌNH CHO LAMBDA = {lam} | SEED = {seed}")
            print(f"################################################################################\n", flush=True)

            sl_out = os.path.join(sweep_root, f"lam{lam}_s{seed}")
            fresh_out = os.path.join(sweep_root, f"fresh_lam{lam}_s{seed}")
            fsha_out = os.path.join(sweep_root, f"fsha_lam{lam}_s{seed}")
            os.makedirs(sl_out, exist_ok=True)
            os.makedirs(fresh_out, exist_ok=True)
            if args.run_fsha:
                os.makedirs(fsha_out, exist_ok=True)

            ckpt_path = os.path.join(sl_out, "checkpoints", f"artape_v0_k{args.k}_linear_fixed_fixR_lam{lam}_s{seed}_client.pt")
            server_ckpt = os.path.join(sl_out, "checkpoints", f"artape_v0_k{args.k}_linear_fixed_fixR_lam{lam}_s{seed}_server.pt")
            sl_json = os.path.join(sl_out, "results_artape_v0.json")

            # ----------------------------------------------------------------------
            # 1. Huấn luyện Split Learning (100 epochs, Cosine Scheduler)
            # ----------------------------------------------------------------------
            sl_done = os.path.isfile(ckpt_path) and os.path.isfile(sl_json)
            if sl_done and not args.force:
                print(f"[RESUME / SKIP] Đã tìm thấy checkpoint client và file kết quả SL tại:\n  -> {ckpt_path}\nBỏ qua 100 epochs huấn luyện Split Learning để tiết kiệm thời gian!\n", flush=True)
            else:
                cmd_sl = [
                    python_bin, "train_artape_v0.py",
                    "--k", str(args.k),
                    "--m", str(args.m),
                    "--unprojector-mode", "linear_fixed",
                    "--fixed-bottleneck",
                    "--lambdas", str(lam),
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

                run_cmd(cmd_sl, desc=f"[1/3] Huấn luyện Split Learning Cosine λ={lam} (Seed {seed})")

            # ----------------------------------------------------------------------
            # 2. Đánh giá Fresh Attackers (MLP và Conv, 30 epochs)
            # ----------------------------------------------------------------------
            fresh_json = os.path.join(fresh_out, "results_artape_v0_fresh_attackers.json")
            fresh_done = False
            if os.path.isfile(fresh_json):
                try:
                    with open(fresh_json, "r", encoding="utf-8") as f:
                        f_chk = json.load(f)
                    atks = [x.get("attacker_type") for x in f_chk]
                    if "mlp" in atks and "conv" in atks:
                        fresh_done = True
                except Exception:
                    fresh_done = False

            if fresh_done and not args.force:
                print(f"[RESUME / SKIP] Cả 2 Fresh Attackers (MLP & Conv) cho λ={lam} (Seed {seed}) đã hoàn tất tại:\n  -> {fresh_json}\nBỏ qua bước đánh giá Fresh!\n", flush=True)
            else:
                cmd_fresh = [
                    python_bin, "eval_artape_v0_fresh.py",
                    "--k", str(args.k),
                    "--fixed-bottleneck",
                    "--client-ckpt", ckpt_path,
                    "--attackers", "mlp", "conv",
                    "--lambdas", str(lam),
                    "--seed", str(seed),
                    "--epochs", str(args.fresh_epochs),
                    "--batch-size", "128",
                    "--lr", "1e-3",
                    "--data-dir", args.data_dir,
                    "--output-dir", fresh_out
                ]
                if args.dry_run:
                    cmd_fresh.append("--dry-run")

                run_cmd(cmd_fresh, desc=f"[2/3] Đánh giá Fresh MLP & Conv λ={lam} (Seed {seed})")

            # ----------------------------------------------------------------------
            # 3. Đánh giá Tấn công Chủ động FSHA (gs=1.0 & gs=5.0, 30 epochs)
            # ----------------------------------------------------------------------
            fsha_json = os.path.join(fsha_out, "results_artape_v0_fsha.json")
            if args.run_fsha:
                fsha_done = False
                if os.path.isfile(fsha_json):
                    try:
                        with open(fsha_json, "r", encoding="utf-8") as f:
                            f_chk = json.load(f)
                        gs_list = [x.get("grad_scale") for x in f_chk]
                        if 1.0 in gs_list and 5.0 in gs_list:
                            fsha_done = True
                    except Exception:
                        fsha_done = False

                if fsha_done and not args.force:
                    print(f"[RESUME / SKIP] Đánh giá FSHA cho λ={lam} (Seed {seed}) đã hoàn tất tại:\n  -> {fsha_json}\nBỏ qua bước FSHA!\n", flush=True)
                else:
                    cmd_fsha = [
                        python_bin, "run_artape_v0_fsha.py",
                        "--k", str(args.k),
                        "--m", str(args.m),
                        "--unprojector-mode", "linear_fixed",
                        "--fixed-bottleneck",
                        "--client-ckpt", ckpt_path,
                        "--server-ckpt", server_ckpt,
                        "--grad-scales", "1.0", "5.0",
                        "--seed", str(seed),
                        "--epochs", str(args.fsha_epochs),
                        "--batch-size", "128",
                        "--data-dir", args.data_dir,
                        "--output-dir", fsha_out
                    ]
                    run_cmd(cmd_fsha, desc=f"[3/3] Đánh giá FSHA Chủ Động gs=1.0 & gs=5.0 λ={lam} (Seed {seed})")

            # ----------------------------------------------------------------------
            # 4. Thu thập kết quả trực tiếp từ JSON
            # ----------------------------------------------------------------------
            acc_10k, acc_2k, dcor = None, None, None
            if os.path.isfile(sl_json):
                with open(sl_json, "r", encoding="utf-8") as f:
                    sl_d = json.load(f)
                key = list(sl_d.keys())[0]
                acc_10k = sl_d[key].get("final_full_test_acc")
                acc_2k = sl_d[key].get("mean_task_acc_last10_2k")
                dcor = sl_d[key].get("final_dcor", sl_d[key].get("history", [{}])[-1].get("dcor"))

            mlp_ssim_10k, conv_ssim_10k = None, None
            if os.path.isfile(fresh_json):
                with open(fresh_json, "r", encoding="utf-8") as f:
                    f_d = json.load(f)
                for item in f_d:
                    if item["attacker_type"] == "mlp":
                        mlp_ssim_10k = item.get("fresh_ssim_10k")
                    elif item["attacker_type"] == "conv":
                        conv_ssim_10k = item.get("fresh_ssim_10k")

            fsha_ssim_gs5, fsha_acc_gs5 = None, None
            if os.path.isfile(fsha_json):
                with open(fsha_json, "r", encoding="utf-8") as f:
                    fsha_d = json.load(f)
                sc5 = next((x for x in fsha_d if x.get("grad_scale") == 5.0), None)
                if sc5:
                    fsha_ssim_gs5 = sc5.get("final_ssim")
                    fsha_acc_gs5 = sc5.get("converged_task_acc_full")

            rec = {
                "lambda": lam,
                "seed": seed,
                "acc_10k": acc_10k,
                "acc_2k": acc_2k,
                "mlp_ssim_10k": mlp_ssim_10k,
                "conv_ssim_10k": conv_ssim_10k,
                "dcor": dcor,
                "fsha_ssim_gs5": fsha_ssim_gs5,
                "fsha_acc_gs5": fsha_acc_gs5
            }
            summary_records.append(rec)

    # --------------------------------------------------------------------------
    # 5. Xuất Báo Cáo Tổng Hợp & In Bảng Đối Chuẩn
    # --------------------------------------------------------------------------
    summary_json_path = os.path.join(sweep_root, "summary_utility_sweep.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 105)
    print("BẢNG TỔNG HỢP KẾT QUẢ UTILITY SWEEP & FSHA AR-TAPE v1 (Cosine, k=2048):")
    print("=" * 105)
    print(f"{'λ':<6} | {'Seed':<6} | {'Acc (10k Full)':<15} | {'Fresh MLP':<12} | {'Fresh Conv':<12} | {'dCor':<8} | {'FSHA SSIM':<11} | {'FSHA Acc'}")
    print("-" * 105)
    for r in summary_records:
        acc10 = f"{r['acc_10k']:.2f}%" if r['acc_10k'] is not None else "N/A"
        mlp_s = f"{r['mlp_ssim_10k']:.4f}" if r['mlp_ssim_10k'] is not None else "N/A"
        conv_s = f"{r['conv_ssim_10k']:.4f}" if r['conv_ssim_10k'] is not None else "N/A"
        dc = f"{r['dcor']:.4f}" if r['dcor'] is not None else "N/A"
        fs_s = f"{r['fsha_ssim_gs5']:.4f}" if r['fsha_ssim_gs5'] is not None else "N/A"
        fs_a = f"{r['fsha_acc_gs5']:.2f}%" if r['fsha_acc_gs5'] is not None else "N/A"
        print(f"{r['lambda']:<6} | {r['seed']:<6} | {acc10:<15} | {mlp_s:<12} | {conv_s:<12} | {dc:<8} | {fs_s:<11} | {fs_a}")
    print("=" * 105)
    print(f"Đã lưu kết quả tại: {summary_json_path}\n")


if __name__ == "__main__":
    main()
