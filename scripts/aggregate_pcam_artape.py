#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/aggregate_pcam_artape.py
==============================================================================
TỔNG HỢP VÀ THỐNG KÊ ĐA SEED CHO M4 AR-TAPE TRÊN PCAM
KÈM BẢNG ĐỐI CHUẨN TOÀN DIỆN (MASTER BENCHMARK: FLOOR VS B0 VS B7 VS AR-TAPE)
==============================================================================
"""

import os
import sys
import glob
import json
import csv
import argparse
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Tổng hợp kết quả M4 AR-TAPE PCAM")
    parser.add_argument("--artape-dir", type=str, default="./output/pcam_m4_artape", help="Thư mục chứa kết quả AR-TAPE")
    parser.add_argument("--b0-dir", type=str, default="./output/pcam_b0_vanilla", help="Thư mục chứa kết quả B0")
    parser.add_argument("--b7-dir", type=str, default="./output/pcam_b7_lightsplit", help="Thư mục chứa kết quả B7")
    parser.add_argument("--floor-json", type=str, default="./output/pcam_benchmark_summary/pcam_reconstruction_floor.json", help="File mốc sàn")
    parser.add_argument("--output-dir", type=str, default="./output/pcam_benchmark_summary", help="Thư mục xuất kết quả")
    return parser.parse_args()


def compute_sample_stats(values):
    vals = np.array(values, dtype=float)
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean_val, std_val


def load_artape_seed_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary_for_poc", {})
        seed = summary.get("seed", data.get("seed"))
        best_epoch = summary.get("best_epoch", data.get("best_epoch", 0))
        best_val_auc = summary.get("best_val_auc", data.get("best_val_auc", 0.0))
        test_auc = summary.get("test_auc", data.get("test_auc_roc", 0.0))
        test_acc = summary.get("test_accuracy", data.get("test_acc", 0.0))
        sens_90 = summary.get("test_sens_at_90_spec", 0.0)

        conv_ssim = summary.get("fresh_conv_ssim", 0.0)
        conv_psnr = summary.get("fresh_conv_psnr", 0.0)
        mlp_ssim = summary.get("fresh_mlp_ssim", 0.0)
        mlp_psnr = summary.get("fresh_mlp_psnr", 0.0)
        fsha1_ssim = summary.get("fsha_gs1_ssim", 0.0)
        fsha1_acc = summary.get("fsha_gs1_task_acc", 0.0)
        fsha5_ssim = summary.get("fsha_gs5_ssim", 0.0)
        fsha5_acc = summary.get("fsha_gs5_task_acc", 0.0)
        git_commit = summary.get("git_commit", "N/A")

        return {
            "file": file_path,
            "seed": int(seed) if seed is not None else -1,
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "test_auc": float(test_auc),
            "test_accuracy": float(test_acc),
            "test_sens_at_90_spec": float(sens_90),
            "fresh_conv_ssim": float(conv_ssim),
            "fresh_conv_psnr": float(conv_psnr),
            "fresh_mlp_ssim": float(mlp_ssim),
            "fresh_mlp_psnr": float(mlp_psnr),
            "fsha_gs1_ssim": float(fsha1_ssim),
            "fsha_gs1_task_acc": float(fsha1_acc),
            "fsha_gs5_ssim": float(fsha5_ssim),
            "fsha_gs5_task_acc": float(fsha5_acc),
            "git_commit": str(git_commit)
        }
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc được file {file_path}: {e}")
        return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Tìm các file JSON AR-TAPE
    patterns = [
        os.path.join(args.artape_dir, "results_pcam_artape_seed_*.json"),
        os.path.join(args.artape_dir, "**", "results_pcam_artape_seed_*.json"),
    ]
    candidate_files = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if p not in candidate_files and not p.endswith("summary.json") and not p.endswith("multiseed.json"):
                candidate_files.append(p)

    if not candidate_files:
        print(f"[LỖI] Không tìm thấy file JSON kết quả AR-TAPE nào trong {args.artape_dir}!")
        sys.exit(1)

    records = []
    seen_seeds = set()
    for fp in sorted(candidate_files):
        rec = load_artape_seed_json(fp)
        if rec and rec["seed"] not in seen_seeds:
            seen_seeds.add(rec["seed"])
            records.append(rec)

    records.sort(key=lambda x: (x["seed"] not in [42, 7, 2024], x["seed"]))
    n_runs = len(records)

    # Tính thống kê mẫu
    seeds = [r["seed"] for r in records]
    m_ep, s_ep = compute_sample_stats([r["best_epoch"] for r in records])
    m_vauc, s_vauc = compute_sample_stats([r["best_val_auc"] for r in records])
    m_tauc, s_tauc = compute_sample_stats([r["test_auc"] for r in records])
    m_acc, s_acc = compute_sample_stats([r["test_accuracy"] for r in records])
    m_s90, s_s90 = compute_sample_stats([r["test_sens_at_90_spec"] * 100.0 for r in records])
    m_cssim, s_cssim = compute_sample_stats([r["fresh_conv_ssim"] for r in records])
    m_cpsnr, s_cpsnr = compute_sample_stats([r["fresh_conv_psnr"] for r in records])
    m_mssim, s_mssim = compute_sample_stats([r["fresh_mlp_ssim"] for r in records])
    m_mpsnr, s_mpsnr = compute_sample_stats([r["fresh_mlp_psnr"] for r in records])
    m_fs1_s, s_fs1_s = compute_sample_stats([r["fsha_gs1_ssim"] for r in records])
    m_fs1_a, s_fs1_a = compute_sample_stats([r["fsha_gs1_task_acc"] for r in records])
    m_fs5_s, s_fs5_s = compute_sample_stats([r["fsha_gs5_ssim"] for r in records])
    m_fs5_a, s_fs5_a = compute_sample_stats([r["fsha_gs5_task_acc"] for r in records])

    print("=" * 110)
    print(f"BÁO CÁO TỔNG HỢP KẾT QUẢ M4 AR-TAPE TRÊN PCAM (k=2048, λ=0.25, N={n_runs})")
    print("=" * 110)

    print("\n### BẢNG KẾT QUẢ TỪNG SEED CHO AR-TAPE (MARKDOWN)\n")
    print("| Seed | Best Ep | Val AUC | Test AUC | Test Acc (%) | Sens@90%Spec (%) | Fresh Conv SSIM | Fresh MLP SSIM | FSHA gs=1 SSIM | FSHA gs=5 SSIM |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in records:
        print(
            f"| `{r['seed']}` | {r['best_epoch']} | {r['best_val_auc']:.4f} | "
            f"**{r['test_auc']:.4f}** | **{r['test_accuracy']:.2f}%** | "
            f"**{r['test_sens_at_90_spec']*100:.2f}%** | "
            f"**{r['fresh_conv_ssim']:.4f}** ({r['fresh_conv_psnr']:.1f}dB) | "
            f"**{r['fresh_mlp_ssim']:.4f}** ({r['fresh_mlp_psnr']:.1f}dB) | "
            f"**{r['fsha_gs1_ssim']:.4f}** ({r['fsha_gs1_task_acc']:.1f}%) | "
            f"**{r['fsha_gs5_ssim']:.4f}** ({r['fsha_gs5_task_acc']:.1f}%) |"
        )
    print(
        f"| **Mean ± Std (n={n_runs})** | **{m_ep:.1f} ± {s_ep:.1f}** | "
        f"**{m_vauc:.4f} ± {s_vauc:.4f}** | **{m_tauc:.4f} ± {s_tauc:.4f}** | "
        f"**{m_acc:.2f} ± {s_acc:.2f}%** | **{m_s90:.2f} ± {s_s90:.2f}%** | "
        f"**{m_cssim:.4f} ± {s_cssim:.4f}** | **{m_mssim:.4f} ± {s_mssim:.4f}** | "
        f"**{m_fs1_s:.4f} ± {s_fs1_s:.4f}** | **{m_fs5_s:.4f} ± {s_fs5_s:.4f}** |"
    )

    # Xuất file JSON
    out_json = os.path.join(args.output_dir, "summary_pcam_artape_multiseed.json")
    summary_data = {
        "dataset": "pcam",
        "method": "AR_TAPE",
        "k": 2048,
        "lambda": 0.25,
        "n_seeds": n_runs,
        "seeds_included": seeds,
        "metrics_summary": {
            "val_auc_mean": round(m_vauc, 4),
            "val_auc_std": round(s_vauc, 4),
            "test_auc_mean": round(m_tauc, 4),
            "test_auc_std": round(s_tauc, 4),
            "test_accuracy_mean": round(m_acc, 2),
            "test_accuracy_std": round(s_acc, 2),
            "test_sens_at_90_spec_pct_mean": round(m_s90, 2),
            "test_sens_at_90_spec_pct_std": round(s_s90, 2),
            "fresh_conv_ssim_mean": round(m_cssim, 4),
            "fresh_conv_ssim_std": round(s_cssim, 4),
            "fresh_mlp_ssim_mean": round(m_mssim, 4),
            "fresh_mlp_ssim_std": round(s_mssim, 4),
            "fsha_gs1_ssim_mean": round(m_fs1_s, 4),
            "fsha_gs1_ssim_std": round(s_fs1_s, 4),
            "fsha_gs5_ssim_mean": round(m_fs5_s, 4),
            "fsha_gs5_ssim_std": round(s_fs5_s, 4),
        },
        "individual_runs": records
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[XUẤT KẾT QUẢ] Đã lưu file tổng hợp AR-TAPE tại: {out_json}")
    print("=" * 110)


if __name__ == "__main__":
    main()
