#!/usr/bin/env python3
# ==============================================================================
# TỰ ĐỘNG TỔNG HỢP VÀ TÍNH TOÁN MEAN ± STD CHO ĐA SEED (42, 7, 2024)
# ==============================================================================
import os
import sys
import glob
import json
import argparse
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Không đọc được file {path}: {e}")
        return None

def find_all_results(search_dirs):
    sl_results = {}     # key: (scheduler, lambda, seed) -> dict
    fresh_results = {}  # key: (scheduler, lambda, seed, attacker) -> dict

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        
        # 1. Tìm results SL
        for f in glob.glob(os.path.join(d, "**", "results_artape_v0.json"), recursive=True):
            data = load_json_file(f)
            if not isinstance(data, dict):
                continue
            for sc_id, item in data.items():
                if not isinstance(item, dict):
                    continue
                lam = item.get("lambda")
                seed = item.get("seed")
                k = item.get("k", 2048)
                unproj = item.get("unprojector_mode", "")
                if lam is not None and seed is not None and k == 2048 and unproj == "linear_fixed":
                    sched = item.get("scheduler") or ("cosine" if "cosine" in f.lower() else "multistep")
                    sl_results[(sched, round(float(lam), 2), int(seed))] = item

        # 2. Tìm results Fresh
        for f in glob.glob(os.path.join(d, "**", "results_artape_v0_fresh_attackers.json"), recursive=True):
            data = load_json_file(f)
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                atk = item.get("attacker_type")
                lam = item.get("lambda")
                seed = item.get("seed")
                if atk and lam is not None and seed is not None:
                    sched = "cosine" if "cosine" in f.lower() else "multistep"
                    fresh_results[(sched, round(float(lam), 2), int(seed), atk)] = item

    return sl_results, fresh_results

def fmt_stat(arr, decimals=2, is_pct=False):
    if len(arr) == 0:
        return "N/A"
    arr = np.array(arr, dtype=float)
    if len(arr) == 1:
        val = f"{arr[0]:.{decimals}f}"
        return f"{val}%" if is_pct else val
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if is_pct:
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}%"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def generate_report(sl_results, fresh_results, target_lambdas=[0.0, 0.5]):
    schedulers = sorted(list(set([k[0] for k in sl_results.keys()])))
    if not schedulers:
        schedulers = ["multistep"]

    for sched in schedulers:
        seeds = sorted(list(set([k[2] for k in sl_results.keys() if k[0] == sched])))
        print("=" * 95)
        print(f"BÁO CÁO TỔNG HỢP ĐA SEED AR-TAPE v1 (Scheduler: {sched.upper()}, Mode F, k=2048)")
        print(f"Các Seed tìm thấy: {seeds} (Số lượng: {len(seeds)})")
        print("=" * 95)

        headers = [
            "Cấu hình", "Seeds", "Full Test Acc (10k)", "Fast Eval Acc (2k)",
            "dCor(X, Z)", "Fresh MLP SSIM", "Fresh MLP PSNR", "Fresh Conv SSIM", "Fresh Conv PSNR"
        ]

        rows = []
        for lam in target_lambdas:
            row_seeds = []
            full_accs, eval_accs, dcors = [], [], []
            mlp_ssims, mlp_psnrs = [], []
            conv_ssims, conv_psnrs = [], []

            for s in [42, 7, 2024]:
                sl = sl_results.get((sched, lam, s))
                if sl:
                    row_seeds.append(str(s))
                    full_accs.append(sl["final_full_test_acc"])
                    eval_accs.append(sl["mean_task_acc_last10_2k"])
                    if sl.get("final_dcor") is not None:
                        dcors.append(sl["final_dcor"])

                f_mlp = fresh_results.get((sched, lam, s, "mlp"))
                if f_mlp:
                    mlp_ssims.append(f_mlp["fresh_ssim_10k"])
                    mlp_psnrs.append(f_mlp["fresh_psnr_10k"])

                f_conv = fresh_results.get((sched, lam, s, "conv"))
                if f_conv:
                    conv_ssims.append(f_conv["fresh_ssim_10k"])
                    conv_psnrs.append(f_conv["fresh_psnr_10k"])

            if not row_seeds:
                continue

            label = f"λ = {lam} ({sched.capitalize()})" + (" [Baseline]" if lam == 0.0 else " [Operating Point]")
            rows.append({
                "config": label,
                "seeds": ",".join(row_seeds) if row_seeds else "Chưa có",
                "full_acc": fmt_stat(full_accs, 2, is_pct=True),
                "eval_acc": fmt_stat(eval_accs, 2, is_pct=True),
                "dcor": fmt_stat(dcors, 4),
                "mlp_ssim": fmt_stat(mlp_ssims, 4),
                "mlp_psnr": fmt_stat(mlp_psnrs, 2) + (" dB" if mlp_psnrs else ""),
                "conv_ssim": fmt_stat(conv_ssims, 4),
                "conv_psnr": fmt_stat(conv_psnrs, 2) + (" dB" if conv_psnrs else "")
            })

        # Render Markdown Table
        md_lines = [
            f"### Bảng Tổng Hợp Đa Seed với Scheduler: {sched.capitalize()}",
            "| Cấu Hình | Seeds Đã Đo | Full Test Acc (10k) | Fast Eval (2k) | $d\\text{Cor}(X, Z)$ | Fresh MLP SSIM | Fresh MLP PSNR | Fresh Conv SSIM | Fresh Conv PSNR |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        for r in rows:
            md_lines.append(f"| **{r['config']}** | {r['seeds']} | **{r['full_acc']}** | {r['eval_acc']} | {r['dcor']} | **{r['mlp_ssim']}** | {r['mlp_psnr']} | **{r['conv_ssim']}** | {r['conv_psnr']} |")

        table_md = "\n".join(md_lines)
        print("\n" + table_md + "\n")

def main():
    parser = argparse.ArgumentParser(description="Tổng hợp kết quả Đa Seed AR-TAPE v1")
    parser.add_argument("--dirs", nargs="+", default=[
        os.path.join(PROJECT_ROOT, "output"),
        os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v1_Utility_lam0"),
        os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v1_MinMax"),
        os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v1_Fresh_Eval"),
        os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v1_MultiSeed"),
        os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v1_MultiSeed_Fresh"),
        os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v1_Cosine"),
        os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v1_Cosine_Fresh"),
        os.path.join(PROJECT_ROOT, "output", "AR_TAPE_v1_Scheduler_Check"),
    ], help="Danh sách thư mục tìm kiếm kết quả")
    args = parser.parse_args()

    sl_res, fresh_res = find_all_results(args.dirs)
    print(f"Đã tìm thấy {len(sl_res)} bản ghi Split Learning và {len(fresh_res)} bản ghi Fresh Attacker.")
    generate_report(sl_res, fresh_res)

if __name__ == "__main__":
    main()
