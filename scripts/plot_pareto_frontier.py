# -*- coding: utf-8 -*-
"""
scripts/plot_pareto_frontier.py
Vẽ biểu đồ Pareto Frontier Trade-off đối chuẩn toàn diện:
Task Accuracy (10k) vs Fresh SSIM (MLP và Conv).
"""
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Cấu hình thẩm mỹ chuẩn publication IEEE/ACM
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

# ==============================================================================
# 1. DỮ LIỆU ĐỐI CHUẨN (10k Full Test)
# ==============================================================================
# B7 LightSplit: k=512, 1024, 2048
b7_acc = [88.54, 91.53, 92.73]
b7_mlp = [0.4394, 0.5461, 0.6865]
b7_conv = [0.2632, 0.3597, 0.5138]
b7_labels = ["B7 (k=512)", "B7 (k=1024)", "B7 (k=2048)"]

# AR-TAPE v1 (Cosine Mode F): lam=0.0 (3 seeds), 0.1 (s42), 0.25 (3 seeds), 0.5 (3 seeds)
artape_lam = [0.0, 0.1, 0.25, 0.5]
artape_acc = [91.05, 90.37, 90.35, 89.41]
artape_acc_err = [0.36, 0.0, 0.24, 0.50]
artape_mlp = [0.7077, 0.2261, 0.2159, 0.2425]
artape_mlp_err = [0.0059, 0.0, 0.0029, 0.0144]
artape_conv = [0.5627, 0.1938, 0.2014, 0.2315]
artape_conv_err = [0.0065, 0.0, 0.0036, 0.0048]

# B8 Stabilized (lam=0.1, cap 1.5586, warmup 10, clip 1.0, s42)
b8_acc = [88.98]
b8_mlp = [0.2621]
b8_conv = [0.6431]

# B0 Vanilla
b0_acc = [94.71]
b0_conv = [0.9955]

# ==============================================================================
# SUBPLOT 1: TASK ACCURACY VS FRESH MLP SSIM
# ==============================================================================
# Vùng an toàn bảo mật (SSIM < 0.30) và Utility cao (Acc >= 90%)
ax1.axvspan(0.0, 0.30, color='green', alpha=0.08, label="Vùng Bảo Mật Khuyến Nghị (SSIM < 0.30)")
ax1.axhspan(90.0, 95.0, color='blue', alpha=0.05, label="Vùng Utility Cao (Acc ≥ 90.0%)")
ax1.axvline(0.30, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
ax1.axhline(90.0, color='navy', linestyle=':', alpha=0.5, linewidth=1.5)

# B7 Curve
ax1.plot(b7_mlp, b7_acc, 's--', color='#d95f02', label="B7 LightSplit (Nén rank k)", zorder=3)
for i, txt in enumerate(b7_labels):
    offset = (-25, 6) if i != 1 else (10, -12)
    ax1.annotate(txt, (b7_mlp[i], b7_acc[i]), textcoords="offset points", xytext=offset,
                 fontweight='bold', color='#d95f02', fontsize=9.5)

# AR-TAPE Points & Curve
# Đường nối min-max lam=0.1 -> 0.25 -> 0.5
artape_minmax_mlp = [artape_mlp[1], artape_mlp[2], artape_mlp[3]]
artape_minmax_acc = [artape_acc[1], artape_acc[2], artape_acc[3]]

ax1.errorbar([artape_mlp[0]], [artape_acc[0]], xerr=[artape_mlp_err[0]], yerr=[artape_acc_err[0]],
             fmt='X', color='gray', markersize=10, capsize=4, label="AR-TAPE (λ=0, Không đối kháng)", zorder=4)
ax1.annotate("AR-TAPE λ=0.0\n(Rò rỉ lớn)", (artape_mlp[0], artape_acc[0]), textcoords="offset points", xytext=(-85, -20),
             color='gray', fontweight='bold', fontsize=9)

ax1.plot(artape_minmax_mlp, artape_minmax_acc, 'o-', color='#1b9e77', linewidth=2.5, label="AR-TAPE Min-Max Frontier", zorder=4)

# Error bar cho các điểm min-max
for j, idx in enumerate([1, 2, 3]):
    if artape_mlp_err[idx] > 0 or artape_acc_err[idx] > 0:
        ax1.errorbar([artape_mlp[idx]], [artape_acc[idx]], xerr=[artape_mlp_err[idx]], yerr=[artape_acc_err[idx]],
                     fmt='none', color='#1b9e77', capsize=3, zorder=4)

# Highlight Sweet Spot λ=0.25
ax1.scatter([artape_mlp[2]], [artape_acc[2]], color='#e7298a', s=160, zorder=5, edgecolors='black', linewidth=1.5)
ax1.annotate("★ ĐIỂM VẬN HÀNH CHÍNH (λ=0.25, 3 seeds)\n(Acc: 90.35 ± 0.24%, MLP SSIM: 0.216)",
             (artape_mlp[2], artape_acc[2]), textcoords="offset points", xytext=(15, 10),
             fontweight='bold', color='#e7298a', fontsize=9.5,
             arrowprops=dict(arrowstyle="->", color='#e7298a', lw=1.5))

ax1.annotate("λ=0.10 (Acc 90.37%, 0.226)", (artape_mlp[1], artape_acc[1]), textcoords="offset points", xytext=(-105, -20),
             fontweight='bold', color='#1b9e77', fontsize=9)
ax1.annotate("λ=0.50 (Acc 89.41%, 0.243)", (artape_mlp[3], artape_acc[3]), textcoords="offset points", xytext=(15, -15),
             fontweight='bold', color='#1b9e77', fontsize=9)

# B8 Stabilized
ax1.plot(b8_mlp, b8_acc, '^', color='#7570b3', markersize=11, label="B8 Stabilized (λ=0.1)", zorder=4)
ax1.annotate("B8 Stabilized\n(Acc 88.98%, 0.262)", (b8_mlp[0], b8_acc[0]), textcoords="offset points", xytext=(10, -18),
             color='#7570b3', fontweight='bold', fontsize=9)

ax1.set_xlabel("Reconstruction SSIM (Fresh MLP 10k) → Càng thấp càng bảo mật")
ax1.set_ylabel("Full Test Accuracy (10k Full Test Set, %) → Càng cao càng tốt")
ax1.set_title("(a) Đánh Đổi Utility vs Kháng Cự Fresh MLP")
ax1.set_xlim(0.10, 0.80)
ax1.set_ylim(87.5, 94.0)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower left', framealpha=0.9)


# ==============================================================================
# SUBPLOT 2: TASK ACCURACY VS FRESH CONV SSIM
# ==============================================================================
ax2.axvspan(0.0, 0.30, color='green', alpha=0.08, label="Vùng Bảo Mật Khuyến Nghị (SSIM < 0.30)")
ax2.axhspan(90.0, 95.0, color='blue', alpha=0.05, label="Vùng Utility Cao (Acc ≥ 90.0%)")
ax2.axvline(0.30, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
ax2.axhline(90.0, color='navy', linestyle=':', alpha=0.5, linewidth=1.5)

# B7 Curve
ax2.plot(b7_conv, b7_acc, 's--', color='#d95f02', label="B7 LightSplit (Nén rank k)", zorder=3)
for i, txt in enumerate(b7_labels):
    offset = (-25, 6) if i != 1 else (10, -12)
    ax2.annotate(txt, (b7_conv[i], b7_acc[i]), textcoords="offset points", xytext=offset,
                 fontweight='bold', color='#d95f02', fontsize=9.5)

# AR-TAPE Points & Curve
artape_minmax_conv = [artape_conv[1], artape_conv[2], artape_conv[3]]
ax2.errorbar([artape_conv[0]], [artape_acc[0]], xerr=[artape_conv_err[0]], yerr=[artape_acc_err[0]],
             fmt='X', color='gray', markersize=10, capsize=4, label="AR-TAPE (λ=0, Không đối kháng)", zorder=4)
ax2.annotate("AR-TAPE λ=0.0\n(Rò rỉ lớn)", (artape_conv[0], artape_acc[0]), textcoords="offset points", xytext=(-85, -20),
             color='gray', fontweight='bold', fontsize=9)

ax2.plot(artape_minmax_conv, artape_minmax_acc, 'o-', color='#1b9e77', linewidth=2.5, label="AR-TAPE Min-Max Frontier", zorder=4)

# Error bar cho các điểm min-max
for j, idx in enumerate([1, 2, 3]):
    if artape_conv_err[idx] > 0 or artape_acc_err[idx] > 0:
        ax2.errorbar([artape_conv[idx]], [artape_acc[idx]], xerr=[artape_conv_err[idx]], yerr=[artape_acc_err[idx]],
                     fmt='none', color='#1b9e77', capsize=3, zorder=4)

# Highlight Sweet Spot λ=0.25
ax2.scatter([artape_conv[2]], [artape_acc[2]], color='#e7298a', s=160, zorder=5, edgecolors='black', linewidth=1.5)
ax2.annotate("★ ĐIỂM VẬN HÀNH CHÍNH (λ=0.25, 3 seeds)\n(Acc: 90.35 ± 0.24%, Conv SSIM: 0.201)",
             (artape_conv[2], artape_acc[2]), textcoords="offset points", xytext=(15, 10),
             fontweight='bold', color='#e7298a', fontsize=9.5,
             arrowprops=dict(arrowstyle="->", color='#e7298a', lw=1.5))

ax2.annotate("λ=0.10 (Conv: 0.194)", (artape_conv[1], artape_acc[1]), textcoords="offset points", xytext=(-95, -15),
             fontweight='bold', color='#1b9e77', fontsize=9)

# B8 Stabilized - Highlight Generalization Failure
ax2.plot(b8_conv, b8_acc, '^', color='#7570b3', markersize=11, label="B8 Stabilized (λ=0.1)", zorder=4)
ax2.annotate("B8 VỠ TRẬN TRƯỚC CONV!\n(SSIM vọt lên 0.643, G=+0.295)",
             (b8_conv[0], b8_acc[0]), textcoords="offset points", xytext=(-120, 18),
             color='#d95f02', fontweight='bold', fontsize=9.5,
             arrowprops=dict(arrowstyle="->", color='#d95f02', lw=1.5))

ax2.set_xlabel("Reconstruction SSIM (Fresh Conv 10k) → Càng thấp càng bảo mật")
ax2.set_ylabel("Full Test Accuracy (10k Full Test Set, %) → Càng cao càng tốt")
ax2.set_title("(b) Đánh Đổi Utility vs Kháng Cự Fresh Conv")
ax2.set_xlim(0.10, 0.80)
ax2.set_ylim(87.5, 94.0)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower left', framealpha=0.9)

plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "pareto_frontier_tradeoff.png")
plt.savefig(out_path, dpi=300)
print(f"Đã lưu biểu đồ Pareto Frontier tại: {out_path}")

# Sao chép sang artifact directory để hiển thị cho người dùng
artifact_dir = r"C:\Users\ADMIN\.gemini\antigravity\brain\c0b937be-df93-44c3-bdbd-06e4f2dd8324"
if os.path.isdir(artifact_dir):
    import shutil
    art_path = os.path.join(artifact_dir, "pareto_frontier_tradeoff.png")
    shutil.copyfile(out_path, art_path)
    print(f"Đã sao chép sang artifact directory tại: {art_path}")
