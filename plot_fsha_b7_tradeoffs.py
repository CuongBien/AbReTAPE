"""
Script ve do thi danh doi (Trade-off) chuan cong bo khoa hoc cho Baseline B7: LightSplit
duoi cuoc tan cong chu dong FSHA (Feature Space Hijacking Attack).

Sinh ra 2 do thi:
1. output/AbReTAPE_FSHA/fsha_b7_tradeoff_by_k.png:
   Khao sat theo so chieu nen k in {512, 1024, 2048} (CR = 128x, 64x, 32x) tai gs = 5.0.
2. output/AbReTAPE_FSHA/fsha_b7_tradeoff_by_gs.png:
   Khao sat do nhay theo he so gradient doi khang gs in {1.0, 2.0, 5.0} tai k = 1024 (CR = 64x).
"""

import os
import sys
import json

# Dam bao encoding ho tro Unicode tren Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def extract_metrics(record):
    # Lay metrics hoi tu (Epoch 30) va peak
    final_h = record["history"][-1]
    conv_acc = final_h.get("test_acc", record.get("test_acc", 0.0)) * 100
    peak_psnr = record.get("psnr", 0.0)
    conv_psnr = final_h.get("test_psnr", peak_psnr)
    peak_ssim = record.get("ssim", 0.0)
    conv_ssim = final_h.get("test_ssim", peak_ssim)
    dcor = record.get("dcor_after", final_h.get("dcor", 0.0))
    return {
        "conv_acc": conv_acc,
        "peak_psnr": peak_psnr,
        "conv_psnr": conv_psnr,
        "peak_ssim": peak_ssim,
        "conv_ssim": conv_ssim,
        "dcor": dcor,
    }

def plot_tradeoff_by_k(data, save_path):
    # Loc cac kich ban voi gs = 5.0
    k_records = {}
    for r in data:
        gs = r.get("grad_scale")
        if gs == 5.0:
            k = r.get("k")
            if k is not None:
                k_records[k] = extract_metrics(r)

    ks = sorted(list(k_records.keys())) # [512, 1024, 2048]
    cr_labels = [f"k={k}\n(CR={65536//k}x)" for k in ks]

    accs = [k_records[k]["conv_acc"] for k in ks]
    peak_psnrs = [k_records[k]["peak_psnr"] for k in ks]
    conv_psnrs = [k_records[k]["conv_psnr"] for k in ks]
    ssims = [k_records[k]["peak_ssim"] for k in ks]
    dcors = [k_records[k]["dcor"] for k in ks]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # (1) Test Accuracy vs k
    ax = axes[0, 0]
    ax.plot(ks, accs, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="B7 Converged Acc (Ep 30)")
    ax.axhline(94.71, color="#2ca02c", linestyle="--", linewidth=1.5, label="B0 Vanilla Acc (94.71%)")
    for x, y in zip(ks, accs):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontweight="bold", fontsize=10, color="#1f77b4")
    ax.set_title("(a) Khả Năng Giữ Tiện Ích (Test Accuracy vs k)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Số chiều chiếu k (Kích thước bottleneck)", fontsize=11)
    ax.set_ylabel("Độ chính xác kiểm thử (%)", fontsize=11)
    ax.set_xticks(ks)
    ax.set_xticklabels(cr_labels)
    ax.set_ylim(50, 100)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    # (2) PSNR vs k
    ax = axes[0, 1]
    ax.plot(ks, peak_psnrs, marker="s", color="#d62728", linewidth=2.5, markersize=8, label="FSHA Peak PSNR")
    ax.plot(ks, conv_psnrs, marker="^", color="#ff7f0e", linestyle=":", linewidth=2.0, markersize=7, label="Converged PSNR (Ep 30)")
    ax.axhline(20.0, color="#b2182b", linestyle="--", linewidth=1.5, label="Ngưỡng sụp đổ (20 dB)")
    ax.fill_between(ks, 0, 20.0, color="#2ca02c", alpha=0.10, label="Vùng bảo vệ an toàn (< 20 dB)")
    for x, y in zip(ks, peak_psnrs):
        ax.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontweight="bold", fontsize=10, color="#d62728")
    ax.set_title("(b) Chất Lượng Tái Tạo FSHA (Peak PSNR vs k)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Số chiều chiếu k (Kích thước bottleneck)", fontsize=11)
    ax.set_ylabel("PSNR (dB - Càng thấp càng an toàn)", fontsize=11)
    ax.set_xticks(ks)
    ax.set_xticklabels(cr_labels)
    ax.set_ylim(8, 25)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    # (3) SSIM vs k
    ax = axes[1, 0]
    ax.plot(ks, ssims, marker="D", color="#9467bd", linewidth=2.5, markersize=8, label="FSHA SSIM")
    ax.axhline(0.40, color="#b2182b", linestyle="--", linewidth=1.5, label="Ngưỡng cấu trúc lộ diện (0.40)")
    ax.fill_between(ks, 0, 0.40, color="#2ca02c", alpha=0.10, label="Vùng mờ cấu trúc hoàn toàn (< 0.40)")
    for x, y in zip(ks, ssims):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontweight="bold", fontsize=10, color="#9467bd")
    ax.set_title("(c) Độ Tương Đồng Cấu Trúc (SSIM vs k)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Số chiều chiếu k (Kích thước bottleneck)", fontsize=11)
    ax.set_ylabel("SSIM (0 -> 1)", fontsize=11)
    ax.set_xticks(ks)
    ax.set_xticklabels(cr_labels)
    ax.set_ylim(0.0, 0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    # (4) dCor vs k
    ax = axes[1, 1]
    ax.plot(ks, dcors, marker="p", color="#8c564b", linewidth=2.5, markersize=9, label="Distance Correlation dCor(X, Z_hat)")
    for x, y in zip(ks, dcors):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontweight="bold", fontsize=10, color="#8c564b")
    ax.set_title("(d) Rò Rỉ Thông Tin Thống Kê (dCor vs k)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Số chiều chiếu k (Kích thước bottleneck)", fontsize=11)
    ax.set_ylabel("Distance Correlation dCor", fontsize=11)
    ax.set_xticks(ks)
    ax.set_xticklabels(cr_labels)
    ax.set_ylim(0.4, 0.8)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Đánh Đổi Tiện Ích vs. Bảo Mật Của Baseline B7: LightSplit Dưới Tấn Công FSHA (grad_scale = 5.0)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Da xuat do thi Trade-off theo k tai: {save_path}")

def plot_tradeoff_by_gs(data, save_path):
    # Lay cac cau hinh k=1024, quet gs in [1.0, 2.0, 5.0]
    gs_records = {}
    for r in data:
        s_id = r.get("scenario_id", "")
        if "k1024" in s_id:
            gs = r.get("grad_scale")
            if gs is not None:
                gs_records[gs] = extract_metrics(r)

    gss = sorted(list(gs_records.keys())) # [1.0, 2.0, 5.0]
    accs = [gs_records[g]["conv_acc"] for g in gss]
    peak_psnrs = [gs_records[g]["peak_psnr"] for g in gss]
    conv_psnrs = [gs_records[g]["conv_psnr"] for g in gss]
    ssims = [gs_records[g]["peak_ssim"] for g in gss]
    dcors = [gs_records[g]["dcor"] for g in gss]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # (1) Test Accuracy vs gs
    ax = axes[0, 0]
    ax.plot(gss, accs, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="B7 Converged Acc (Ep 30)")
    ax.axhline(91.53, color="#2ca02c", linestyle="--", linewidth=1.5, label="B7 Passive Acc (k=1024, 91.53%)")
    for x, y in zip(gss, accs):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontweight="bold", fontsize=10, color="#1f77b4")
    ax.set_title("(a) Độ Nhạy Tiện Ích Theo Cường Độ Gradient (Acc vs grad_scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hệ số khuếch đại gradient đối kháng (grad_scale)", fontsize=11)
    ax.set_ylabel("Độ chính xác kiểm thử (%)", fontsize=11)
    ax.set_xticks(gss)
    ax.set_ylim(50, 100)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    # (2) PSNR vs gs
    ax = axes[0, 1]
    ax.plot(gss, peak_psnrs, marker="s", color="#d62728", linewidth=2.5, markersize=8, label="FSHA Peak PSNR")
    ax.plot(gss, conv_psnrs, marker="^", color="#ff7f0e", linestyle=":", linewidth=2.0, markersize=7, label="Converged PSNR (Ep 30)")
    ax.axhline(20.0, color="#b2182b", linestyle="--", linewidth=1.5, label="Ngưỡng sụp đổ (20 dB)")
    ax.fill_between(gss, 0, 20.0, color="#2ca02c", alpha=0.10, label="Vùng bảo vệ an toàn (< 20 dB)")
    for x, y in zip(gss, peak_psnrs):
        ax.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontweight="bold", fontsize=10, color="#d62728")
    ax.set_title("(b) Độ Nhạy Tái Tạo Ảnh (PSNR vs grad_scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hệ số khuếch đại gradient đối kháng (grad_scale)", fontsize=11)
    ax.set_ylabel("PSNR (dB - Càng thấp càng an toàn)", fontsize=11)
    ax.set_xticks(gss)
    ax.set_ylim(8, 25)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    # (3) SSIM vs gs
    ax = axes[1, 0]
    ax.plot(gss, ssims, marker="D", color="#9467bd", linewidth=2.5, markersize=8, label="FSHA SSIM")
    ax.axhline(0.40, color="#b2182b", linestyle="--", linewidth=1.5, label="Ngưỡng cấu trúc lộ diện (0.40)")
    ax.fill_between(gss, 0, 0.40, color="#2ca02c", alpha=0.10, label="Vùng mờ cấu trúc hoàn toàn (< 0.40)")
    for x, y in zip(gss, ssims):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontweight="bold", fontsize=10, color="#9467bd")
    ax.set_title("(c) Độ Tương Đồng Cấu Trúc (SSIM vs grad_scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hệ số khuếch đại gradient đối kháng (grad_scale)", fontsize=11)
    ax.set_ylabel("SSIM (0 -> 1)", fontsize=11)
    ax.set_xticks(gss)
    ax.set_ylim(0.0, 0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    # (4) dCor vs gs
    ax = axes[1, 1]
    ax.plot(gss, dcors, marker="p", color="#8c564b", linewidth=2.5, markersize=9, label="Distance Correlation dCor")
    for x, y in zip(gss, dcors):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontweight="bold", fontsize=10, color="#8c564b")
    ax.set_title("(d) Rò Rỉ Thông Tin Thống Kê (dCor vs grad_scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hệ số khuếch đại gradient đối kháng (grad_scale)", fontsize=11)
    ax.set_ylabel("Distance Correlation dCor", fontsize=11)
    ax.set_xticks(gss)
    ax.set_ylim(0.4, 0.8)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Độ Nhạy Của B7 LightSplit Đối Với Gradient Đối Kháng FSHA (k = 1024, CR = 64x)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Da xuat do thi do nhay theo gs tai: {save_path}")

def main():
    json_path = os.path.join("output", "AbReTAPE_FSHA", "results_fsha_b7.json")
    if not os.path.isfile(json_path):
        print(f"[ERROR] Khong tim thay file du lieu: {json_path}")
        return

    data = load_data(json_path)
    out_dir = os.path.join("output", "AbReTAPE_FSHA")

    plot_k_path = os.path.join(out_dir, "fsha_b7_tradeoff_by_k.png")
    plot_gs_path = os.path.join(out_dir, "fsha_b7_tradeoff_by_gs.png")

    plot_tradeoff_by_k(data, plot_k_path)
    plot_tradeoff_by_gs(data, plot_gs_path)
    print("\n[HOAN TAT] Ca 2 bieu do trade-off da duoc xuat thanh cong!")

if __name__ == "__main__":
    main()
