#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/aggregate_pcam_b0.py
==============================================================================
TỔNG HỢP VÀ THỐNG KÊ KẾT QUẢ HUẤN LUYỆN B0 (VANILLA) TRÊN PCAM QUA 3 SEEDS
==============================================================================
Quy tắc kết quả:
- Chỉ báo cáo trên tập Test tại checkpoint được chọn bằng AUC Validation cao nhất.
- Báo cáo 4 chỉ số cốt lõi:
  1. Best Epoch & Best Val AUC
  2. Test AUC-ROC
  3. Test Accuracy (%)
  4. Test Sensitivity @ 90% Specificity (%)
- Tính toán: Mean ± Sample Std (độ lệch chuẩn mẫu, mẫu số chia n - 1, ddof=1).
- Xuất bảng Markdown và file tổng kết JSON (pcam_b0_summary.json).
==============================================================================
"""

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


def parse_args():
    parser = argparse.ArgumentParser(description="Tổng hợp kết quả đa seed cho PCAM B0")
    parser.add_argument("--results-dir", type=str, default="./output/pcam_b0_vanilla", help="Thư mục chứa kết quả các seed")
    parser.add_argument("--files", nargs="*", default=None, help="Danh sách cụ thể các file JSON kết quả cần tổng hợp")
    parser.add_argument("--output-json", type=str, default=None, help="Đường dẫn lưu file JSON tổng hợp")
    return parser.parse_args()


def load_seed_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Hỗ trợ cả cấu trúc summary_for_poc lẫn cấu trúc phẳng
        summary = data.get("summary_for_poc", {})
        seed = summary.get("seed", data.get("seed"))
        best_epoch = summary.get("best_epoch", data.get("best_epoch"))
        best_val_auc = summary.get("best_val_auc", data.get("best_val_auc"))
        test_auc = summary.get("test_auc", data.get("test_auc_roc", data.get("test_auc")))
        test_acc = summary.get("test_accuracy", data.get("test_acc", data.get("test_accuracy")))
        
        test_clin = data.get("clinical_metrics", {})
        sens_90 = summary.get(
            "test_sens_at_90_spec",
            data.get("test_sens_at_90_spec", test_clin.get("sensitivity_at_90_specificity", 0.0))
        )
        train_time = summary.get("train_time_formatted", data.get("train_time_formatted", "N/A"))
        git_commit = summary.get("git_commit", data.get("git_commit", "N/A"))

        return {
            "file": file_path,
            "seed": int(seed) if seed is not None else -1,
            "best_epoch": int(best_epoch) if best_epoch is not None else 0,
            "best_val_auc": float(best_val_auc) if best_val_auc is not None else 0.0,
            "test_auc": float(test_auc) if test_auc is not None else 0.0,
            "test_accuracy": float(test_acc) if test_acc is not None else 0.0,
            "test_sens_at_90_spec": float(sens_90) if sens_90 is not None else 0.0,
            "train_time": str(train_time),
            "git_commit": str(git_commit)
        }
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể đọc file {file_path}: {e}")
        return None


def compute_sample_stats(values):
    """Tính Mean và Sample Std (chia cho n - 1 với ddof=1)."""
    vals = np.array(values, dtype=float)
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean_val, std_val


def main():
    args = parse_args()

    # Tìm các file JSON kết quả
    candidate_files = []
    if args.files:
        candidate_files = args.files
    else:
        # Tìm đệ quy trong thư mục results-dir
        patterns = [
            os.path.join(args.results_dir, "results_pcam_b0_seed_*.json"),
            os.path.join(args.results_dir, "**", "results_pcam_b0_seed_*.json"),
            os.path.join(args.results_dir, "**", "results_pcam_b0.json")
        ]
        for pat in patterns:
            for p in glob.glob(pat, recursive=True):
                if p not in candidate_files and not p.endswith("summary.json") and not p.endswith("3seeds_summary.json"):
                    candidate_files.append(p)

    if not candidate_files:
        print(f"[LỖI] Không tìm thấy file JSON kết quả nào trong: {args.results_dir}")
        print("Vui lòng kiểm tra lại đường dẫn hoặc truyền danh sách file qua --files.")
        sys.exit(1)

    loaded_records = []
    seen_seeds = set()
    for fp in sorted(candidate_files):
        rec = load_seed_json(fp)
        if rec and rec["seed"] not in seen_seeds:
            seen_seeds.add(rec["seed"])
            loaded_records.append(rec)

    # Sắp xếp theo thứ tự seed chuẩn: [42, 7, 2024] nếu có
    loaded_records.sort(key=lambda x: (x["seed"] not in [42, 7, 2024], x["seed"]))

    n_runs = len(loaded_records)
    print("=" * 100)
    print(f"BÁO CÁO TỔNG HỢP KẾT QUẢ ĐỐI CHUẨN PCAM B0 VANILLA (SỐ LẦN CHẠY N = {n_runs})")
    print("Tiêu chuẩn chọn checkpoint: Tối ưu theo AUC-ROC trên tập Validation")
    print("Độ lệch chuẩn: Sample Standard Deviation (chia n - 1, ddof=1)")
    print("=" * 100)

    # Trích xuất các chuỗi số liệu
    seeds = [r["seed"] for r in loaded_records]
    best_epochs = [r["best_epoch"] for r in loaded_records]
    val_aucs = [r["best_val_auc"] for r in loaded_records]
    test_aucs = [r["test_auc"] for r in loaded_records]
    test_accs = [r["test_accuracy"] for r in loaded_records]
    test_sens90s = [r["test_sens_at_90_spec"] for r in loaded_records]
    test_sens90_pcts = [s * 100.0 for s in test_sens90s]

    m_ep, s_ep = compute_sample_stats(best_epochs)
    m_vauc, s_vauc = compute_sample_stats(val_aucs)
    m_tauc, s_tauc = compute_sample_stats(test_aucs)
    m_acc, s_acc = compute_sample_stats(test_accs)
    m_s90, s_s90 = compute_sample_stats(test_sens90_pcts)

    # In Bảng Markdown
    print("\n### BẢNG KẾT QUẢ TỪNG SEED & TỔNG HỢP (MARKDOWN)\n")
    print("| Seed | Best Epoch | Val AUC | Test AUC | Test Accuracy (%) | Sens@90%Spec (%) | Thời gian | Commit |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in loaded_records:
        print(
            f"| `{r['seed']}` | {r['best_epoch']} | {r['best_val_auc']:.4f} | "
            f"**{r['test_auc']:.4f}** | **{r['test_accuracy']:.2f}%** | "
            f"**{r['test_sens_at_90_spec']*100:.2f}%** | {r['train_time']} | `{r['git_commit'][:7]}` |"
        )
    print(
        f"| **Mean ± Std (n={n_runs})** | **{m_ep:.1f} ± {s_ep:.1f}** | "
        f"**{m_vauc:.4f} ± {s_vauc:.4f}** | **{m_tauc:.4f} ± {s_tauc:.4f}** | "
        f"**{m_acc:.2f} ± {s_acc:.2f}%** | **{m_s90:.2f} ± {s_s90:.2f}%** | - | - |"
    )

    print("\n" + "-" * 100)
    print("CHUỖI ĐỊNH DẠNG SẴN SÀNG CHO BẢNG POC & BÁO CÁO BÀI BÁO:")
    print("-" * 100)
    print(f"Val AUC-ROC:             {m_vauc:.4f} ± {s_vauc:.4f}")
    print(f"Test AUC-ROC:            {m_tauc:.4f} ± {s_tauc:.4f}")
    print(f"Test Accuracy:           {m_acc:.2f}% ± {s_acc:.2f}%")
    print(f"Sensitivity @ 90% Spec:  {m_s90:.2f}% ± {s_s90:.2f}%")
    print("-" * 100)

    # Lưu kết quả tổng hợp ra file JSON
    out_json_path = args.output_json or os.path.join(args.results_dir, "summary_pcam_b0_multiseed.json")
    summary_data = {
        "dataset": "pcam",
        "method": "B0_Vanilla",
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
            "best_epoch_mean": round(m_ep, 1),
            "best_epoch_std": round(s_ep, 1)
        },
        "formatted_text": {
            "val_auc": f"{m_vauc:.4f} ± {s_vauc:.4f}",
            "test_auc": f"{m_tauc:.4f} ± {s_tauc:.4f}",
            "test_accuracy": f"{m_acc:.2f} ± {s_acc:.2f}%",
            "test_sens_at_90_spec": f"{m_s90:.2f} ± {s_s90:.2f}%"
        },
        "individual_runs": loaded_records
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json_path)), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[XUẤT KẾT QUẢ] Đã lưu file tổng hợp tại: {out_json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
