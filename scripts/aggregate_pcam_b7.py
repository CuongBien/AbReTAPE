#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/aggregate_pcam_b7.py
==============================================================================
TỔNG HỢP VÀ THỐNG KÊ ĐA SEED CHO B7 LIGHTSPLIT TRÊN PCAM
KÈM BẢNG ĐỐI CHUẨN SO SÁNH TRỰC TIẾP VỚI B0 VANILLA
==============================================================================
Quy tắc thống kê:
- Toàn bộ chỉ số Test đánh giá tại checkpoint tối ưu theo Validation AUC.
- Attacker metrics đánh giá tại epoch cuối (30 epochs, 50k train).
- Tính toán: Mean ± Sample Std (mẫu số chia n - 1, ddof=1).
- Xuất bảng Markdown và file tổng kết JSON (summary_pcam_b7_multiseed.json).
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
    parser = argparse.ArgumentParser(description="Tổng hợp kết quả đa seed cho PCAM B7 LightSplit")
    parser.add_argument("--b7-dir", type=str, default="./output/pcam_b7_k2048", help="Thư mục chứa kết quả B7")
    parser.add_argument("--b0-dir", type=str, default=None, help="Thư mục chứa kết quả B0 (tự động nạp summary hoặc các seed)")
    parser.add_argument("--b0-summary", type=str, default=None, help="File JSON tổng hợp B0 (summary_pcam_b0_multiseed.json)")
    parser.add_argument("--b0-attacks-dir", "--b0-m2-dir", dest="b0_attacks_dir", type=str, default=None, help="Thư mục chứa kết quả M2 B0 attacks")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục xuất kết quả tổng hợp")
    parser.add_argument("--output-json", type=str, default=None, help="Đường dẫn file JSON tổng hợp")
    return parser.parse_args()


def load_b7_seed_json(file_path):
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
            "git_commit": str(git_commit)
        }
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc được file {file_path}: {e}")
        return None


def load_b0_seed_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary_for_poc", {})
        seed = summary.get("seed", data.get("seed"))
        best_epoch = summary.get("best_epoch", data.get("best_epoch", 0))
        best_val_auc = summary.get("best_val_auc", data.get("best_val_auc", 0.0))
        test_auc = summary.get("test_auc", data.get("test_auc_roc", data.get("test_auc", 0.0)))
        test_acc = summary.get("test_accuracy", data.get("test_acc", data.get("test_accuracy", 0.0)))
        test_clin = data.get("clinical_metrics", {})
        sens_90 = summary.get(
            "test_sens_at_90_spec",
            data.get("test_sens_at_90_spec", test_clin.get("sensitivity_at_90_specificity", 0.0))
        )
        return {
            "file": file_path,
            "seed": int(seed) if seed is not None else -1,
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "test_auc": float(test_auc),
            "test_accuracy": float(test_acc),
            "test_sens_at_90_spec": float(sens_90)
        }
    except Exception:
        return None


def compute_sample_stats(values):
    """Tính Mean và Sample Std (chia n - 1, ddof=1)."""
    vals = np.array(values, dtype=float)
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean_val, std_val


def main():
    args = parse_args()

    # 1. Tìm các file JSON kết quả B7
    patterns = [
        os.path.join(args.b7_dir, "results_pcam_b7_seed_*.json"),
        os.path.join(args.b7_dir, "**", "results_pcam_b7_seed_*.json"),
        os.path.join(args.b7_dir, "**", "results_pcam_b7.json"),
        os.path.join(args.b7_dir, "results_pcam_b7.json")
    ]
    candidate_files = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if p not in candidate_files and not p.endswith("summary.json") and not p.endswith("multiseed.json"):
                candidate_files.append(p)

    if not candidate_files:
        print(f"[LỖI] Không tìm thấy file JSON kết quả B7 nào trong {args.b7_dir}!")
        print("Vui lòng kiểm tra lại đường dẫn --b7-dir.")
        sys.exit(1)

    records = []
    seen_seeds = set()
    for fp in sorted(candidate_files):
        rec = load_b7_seed_json(fp)
        if rec and rec["seed"] not in seen_seeds:
            seen_seeds.add(rec["seed"])
            records.append(rec)

    records.sort(key=lambda x: (x["seed"] not in [42, 7, 2024], x["seed"]))
    n_runs = len(records)

    print("=" * 110)
    print(f"BÁO CÁO TỔNG HỢP KẾT QUẢ ĐỐI CHUẨN PCAM B7 LIGHTSPLIT (k=2048, MODE F, N={n_runs})")
    print("Tiêu chuẩn chọn checkpoint: Tối ưu theo AUC-ROC trên tập Validation")
    print("Độ lệch chuẩn: Sample Standard Deviation (chia n - 1, ddof=1)")
    print("=" * 110)

    seeds = [r["seed"] for r in records]
    best_epochs = [r["best_epoch"] for r in records]
    val_aucs = [r["best_val_auc"] for r in records]
    test_aucs = [r["test_auc"] for r in records]
    test_accs = [r["test_accuracy"] for r in records]
    test_s90s = [r["test_sens_at_90_spec"] * 100.0 for r in records]
    conv_ssims = [r["fresh_conv_ssim"] for r in records]
    conv_psnrs = [r["fresh_conv_psnr"] for r in records]
    mlp_ssims = [r["fresh_mlp_ssim"] for r in records]
    mlp_psnrs = [r["fresh_mlp_psnr"] for r in records]

    m_ep, s_ep = compute_sample_stats(best_epochs)
    m_vauc, s_vauc = compute_sample_stats(val_aucs)
    m_tauc, s_tauc = compute_sample_stats(test_aucs)
    m_acc, s_acc = compute_sample_stats(test_accs)
    m_s90, s_s90 = compute_sample_stats(test_s90s)
    m_cssim, s_cssim = compute_sample_stats(conv_ssims)
    m_cpsnr, s_cpsnr = compute_sample_stats(conv_psnrs)
    m_mssim, s_mssim = compute_sample_stats(mlp_ssims)
    m_mpsnr, s_mpsnr = compute_sample_stats(mlp_psnrs)

    print("\n### BẢNG KẾT QUẢ TỪNG SEED CHO B7 LIGHTSPLIT (MARKDOWN)\n")
    print("| Seed | Best Ep | Val AUC | Test AUC | Test Acc (%) | Sens@90%Spec (%) | Fresh Conv SSIM | Fresh MLP SSIM | Commit |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in records:
        print(
            f"| `{r['seed']}` | {r['best_epoch']} | {r['best_val_auc']:.4f} | "
            f"**{r['test_auc']:.4f}** | **{r['test_accuracy']:.2f}%** | "
            f"**{r['test_sens_at_90_spec']*100:.2f}%** | "
            f"**{r['fresh_conv_ssim']:.4f}** ({r['fresh_conv_psnr']:.1f}dB) | "
            f"**{r['fresh_mlp_ssim']:.4f}** ({r['fresh_mlp_psnr']:.1f}dB) | `{r['git_commit'][:7]}` |"
        )
    print(
        f"| **Mean ± Std (n={n_runs})** | **{m_ep:.1f} ± {s_ep:.1f}** | "
        f"**{m_vauc:.4f} ± {s_vauc:.4f}** | **{m_tauc:.4f} ± {s_tauc:.4f}** | "
        f"**{m_acc:.2f} ± {s_acc:.2f}%** | **{m_s90:.2f} ± {s_s90:.2f}%** | "
        f"**{m_cssim:.4f} ± {s_cssim:.4f}** | **{m_mssim:.4f} ± {s_mssim:.4f}** | - |"
    )

    # 2. Đọc kết quả B0 để tạo bảng đối chiếu B0 vs B7
    b0_summary = None
    if args.b0_summary and os.path.isfile(args.b0_summary):
        try:
            with open(args.b0_summary, "r", encoding="utf-8") as f:
                b0_summary = json.load(f)
        except Exception:
            pass

    if b0_summary is None and args.b0_dir:
        cand_summaries = [
            os.path.join(args.b0_dir, "summary_pcam_b0_multiseed.json"),
            os.path.join(args.b0_dir, "pcam_b0_summary.json"),
            os.path.join(args.b0_dir, "**", "summary_pcam_b0_multiseed.json"),
        ]
        for cs in cand_summaries:
            matches = glob.glob(cs, recursive=True) if "**" in cs else ([cs] if os.path.isfile(cs) else [])
            for m in matches:
                if os.path.isfile(m):
                    try:
                        with open(m, "r", encoding="utf-8") as f:
                            b0_summary = json.load(f)
                            print(f"[THÔNG TIN] Đã tìm thấy file tổng hợp B0: {m}")
                            break
                    except Exception:
                        pass
            if b0_summary is not None:
                break

    # Nếu chưa có summary nhưng có thư mục b0_dir, tổng hợp trực tiếp từ các file seed
    if b0_summary is None and args.b0_dir and os.path.isdir(args.b0_dir):
        b0_patterns = [
            os.path.join(args.b0_dir, "results_pcam_b0_seed_*.json"),
            os.path.join(args.b0_dir, "**", "results_pcam_b0_seed_*.json"),
            os.path.join(args.b0_dir, "**", "results_pcam_b0.json"),
        ]
        b0_files = []
        for pat in b0_patterns:
            for p in glob.glob(pat, recursive=True):
                if p not in b0_files and not p.endswith("summary.json") and not p.endswith("multiseed.json"):
                    b0_files.append(p)
        if b0_files:
            b0_recs = []
            for fp in sorted(b0_files):
                rec = load_b0_seed_json(fp)
                if rec:
                    b0_recs.append(rec)
            if b0_recs:
                print(f"[THÔNG TIN] Tự động tổng hợp {len(b0_recs)} runs B0 từ: {args.b0_dir}")
                b0_accs = [r["test_accuracy"] for r in b0_recs]
                b0_aucs = [r["test_auc"] for r in b0_recs]
                b0_s90s = [r["test_sens_at_90_spec"] * 100.0 for r in b0_recs]
                m_b0_acc, s_b0_acc = compute_sample_stats(b0_accs)
                m_b0_auc, s_b0_auc = compute_sample_stats(b0_aucs)
                m_b0_s90, s_b0_s90 = compute_sample_stats(b0_s90s)
                b0_summary = {
                    "metrics_summary": {
                        "test_accuracy_mean": m_b0_acc,
                        "test_accuracy_std": s_b0_acc,
                        "test_auc_mean": m_b0_auc,
                        "test_auc_std": s_b0_auc,
                        "test_sens_at_90_spec_pct_mean": m_b0_s90,
                        "test_sens_at_90_spec_pct_std": s_b0_s90,
                    }
                }

    # 3. Đọc kết quả M2 B0 attacks nếu có
    b0_attacks = []
    atk_dir = args.b0_attacks_dir or (os.path.join(args.b0_dir, "pcam_m2_b0_attacks") if args.b0_dir else None)
    if atk_dir and os.path.isdir(atk_dir):
        atk_patterns = [
            os.path.join(atk_dir, "results_pcam_b0_attacks_seed_*.json"),
            os.path.join(atk_dir, "**", "results_pcam_b0_attacks_seed_*.json"),
            os.path.join(atk_dir, "**", "results_pcam_b0_attacks.json")
        ]
        seen_atk_files = set()
        for pat in atk_patterns:
            for fp in glob.glob(pat, recursive=True):
                if fp not in seen_atk_files:
                    seen_atk_files.add(fp)
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            summ = data.get("summary_for_poc", {})
                            if summ and ("fresh_conv_ssim" in summ or "fresh_mlp_ssim" in summ):
                                b0_attacks.append(summ)
                    except Exception:
                        pass
        if b0_attacks:
            print(f"[THÔNG TIN] Đã tìm thấy {len(b0_attacks)} kết quả M2 Fresh Attackers cho B0.")

    # 4. Hiển thị Bảng đối chiếu B0 vs B7
    comparison_table_lines = []
    if b0_summary is not None:
        b0_m = b0_summary.get("metrics_summary", {})
        print("\n" + "-" * 110)
        print("BẢNG SO SÁNH TRỰC DIỆN: B0 (VANILLA) VS B7 (LIGHTSPLIT k=2048, MODE F) TRÊN PCAM")
        print("-" * 110)
        header = "| Phương Pháp | Dimension Cut-layer | Test Accuracy (%) | Test AUC-ROC | Sens @ 90% Spec (%) | Fresh Conv SSIM | Fresh MLP SSIM |"
        separator = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
        print(header)
        print(separator)
        comparison_table_lines.extend([header, separator])

        # B0 values
        b0_acc_str = f"{b0_m.get('test_accuracy_mean', 0.0):.2f} ± {b0_m.get('test_accuracy_std', 0.0):.2f}%"
        b0_auc_str = f"{b0_m.get('test_auc_mean', 0.0):.4f} ± {b0_m.get('test_auc_std', 0.0):.4f}"
        b0_s90_str = f"{b0_m.get('test_sens_at_90_spec_pct_mean', 0.0):.2f} ± {b0_m.get('test_sens_at_90_spec_pct_std', 0.0):.2f}%"

        b0_cssim_str = "---"
        b0_mssim_str = "---"
        if b0_attacks:
            c_ssims = [a.get("fresh_conv_ssim", 0.0) for a in b0_attacks if "fresh_conv_ssim" in a]
            m_ssims = [a.get("fresh_mlp_ssim", 0.0) for a in b0_attacks if "fresh_mlp_ssim" in a]
            if c_ssims:
                mc, sc = compute_sample_stats(c_ssims)
                b0_cssim_str = f"{mc:.4f} ± {sc:.4f}"
            if m_ssims:
                mm, sm = compute_sample_stats(m_ssims)
                b0_mssim_str = f"{mm:.4f} ± {sm:.4f}"

        row_b0 = f"| **B0: Vanilla SL** | $D = 65,536$ | {b0_acc_str} | {b0_auc_str} | {b0_s90_str} | {b0_cssim_str} | {b0_mssim_str} |"
        row_b7 = f"| **B7: LightSplit (Mode F)** | $k = 2,048$ | **{m_acc:.2f} ± {s_acc:.2f}%** | **{m_tauc:.4f} ± {s_tauc:.4f}** | **{m_s90:.2f} ± {s_s90:.2f}%** | **{m_cssim:.4f} ± {s_cssim:.4f}** | **{m_mssim:.4f} ± {s_mssim:.4f}** |"

        print(row_b0)
        print(row_b7)
        print("-" * 110)
        comparison_table_lines.extend([row_b0, row_b7])

    # 5. Lưu file JSON & Markdown tổng hợp
    if args.output_dir:
        out_dir = args.output_dir
    elif args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
    else:
        out_dir = args.b7_dir

    os.makedirs(out_dir, exist_ok=True)
    out_json = args.output_json or os.path.join(out_dir, "summary_pcam_b7_multiseed.json")
    out_md = os.path.join(out_dir, "pcam_comparison_b0_vs_b7.md")
    out_csv = os.path.join(out_dir, "pcam_comparison_b0_vs_b7.csv")

    summary_data = {
        "dataset": "pcam",
        "method": "B7_LightSplit",
        "k": 2048,
        "mode": "F",
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
            "fresh_conv_psnr_mean": round(m_cpsnr, 2),
            "fresh_conv_psnr_std": round(s_cpsnr, 2),
            "fresh_mlp_ssim_mean": round(m_mssim, 4),
            "fresh_mlp_ssim_std": round(s_mssim, 4),
            "fresh_mlp_psnr_mean": round(m_mpsnr, 2),
            "fresh_mlp_psnr_std": round(s_mpsnr, 2),
            "best_epoch_mean": round(m_ep, 1),
            "best_epoch_std": round(s_ep, 1)
        },
        "formatted_text": {
            "val_auc": f"{m_vauc:.4f} ± {s_vauc:.4f}",
            "test_auc": f"{m_tauc:.4f} ± {s_tauc:.4f}",
            "test_accuracy": f"{m_acc:.2f} ± {s_acc:.2f}%",
            "test_sens_at_90_spec": f"{m_s90:.2f} ± {s_s90:.2f}%",
            "fresh_conv_ssim": f"{m_cssim:.4f} ± {s_cssim:.4f}",
            "fresh_mlp_ssim": f"{m_mssim:.4f} ± {s_mssim:.4f}"
        },
        "individual_runs": records
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Xuất file Markdown nếu có bảng so sánh
    if comparison_table_lines:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# BÁO CÁO ĐỐI CHUẨN PCAM: B0 VANILLA VS B7 LIGHTSPLIT\n\n")
            f.write("\n".join(comparison_table_lines) + "\n\n")
            f.write("### Chi tiết từng Seed của B7 LightSplit:\n\n")
            f.write("| Seed | Best Ep | Val AUC | Test AUC | Test Acc (%) | Sens@90%Spec (%) | Fresh Conv SSIM | Fresh MLP SSIM |\n")
            f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for r in records:
                f.write(
                    f"| `{r['seed']}` | {r['best_epoch']} | {r['best_val_auc']:.4f} | "
                    f"**{r['test_auc']:.4f}** | **{r['test_accuracy']:.2f}%** | "
                    f"**{r['test_sens_at_90_spec']*100:.2f}%** | "
                    f"**{r['fresh_conv_ssim']:.4f}** ({r['fresh_conv_psnr']:.1f}dB) | "
                    f"**{r['fresh_mlp_ssim']:.4f}** ({r['fresh_mlp_psnr']:.1f}dB) |\n"
                )

    # Xuất file CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Seed", "Best_Epoch", "Val_AUC", "Test_AUC", "Test_Acc", "Sens_at_90_Spec", "Fresh_Conv_SSIM", "Fresh_Conv_PSNR", "Fresh_MLP_SSIM", "Fresh_MLP_PSNR", "Git_Commit"])
        for r in records:
            writer.writerow([
                r["seed"], r["best_epoch"], f"{r['best_val_auc']:.4f}", f"{r['test_auc']:.4f}",
                f"{r['test_accuracy']:.2f}", f"{r['test_sens_at_90_spec']*100:.2f}",
                f"{r['fresh_conv_ssim']:.4f}", f"{r['fresh_conv_psnr']:.2f}",
                f"{r['fresh_mlp_ssim']:.4f}", f"{r['fresh_mlp_psnr']:.2f}",
                r["git_commit"]
            ])

    print(f"\n[XUẤT KẾT QUẢ] Đã lưu file tổng hợp tại:")
    print(f"  - JSON:     {out_json}")
    if comparison_table_lines:
        print(f"  - Markdown: {out_md}")
    print(f"  - CSV:      {out_csv}")
    print("=" * 110)


if __name__ == "__main__":
    main()
