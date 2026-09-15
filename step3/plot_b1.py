# Bước 3 — Trực quan hóa đường cong Trade-off (Accuracy vs PSNR/SSIM) và lưới ảnh tái tạo
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP1_DIR = os.path.join(PROJECT_ROOT, "step1")

for path in [SCRIPT_DIR, PROJECT_ROOT, STEP1_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from metrics import denormalize


def plot_tradeoff(results, save_path=None, show=False):
    """
    Vẽ đồ thị Privacy-Utility Trade-off across different sigmas:
    - Trục hoành: sigma (Mức độ nhiễu)
    - Trục tung trái: Accuracy (Utility)
    - Trục tung phải: PSNR / SSIM / LPIPS (Security)
    results: List các dict, mỗi dict chứa:
      {"sigma": float, "test_acc": float, "psnr": float, "ssim": float, "lpips": float}
    """
    if not results:
        return

    # Sắp xếp theo sigma tăng dần
    sorted_res = sorted(results, key=lambda d: d["sigma"])
    sigmas = [d["sigma"] for d in sorted_res]
    accs = [d["test_acc"] * 100.0 for d in sorted_res]
    psnrs = [d["psnr"] for d in sorted_res]
    ssims = [d["ssim"] for d in sorted_res]
    has_lpips = any(d.get("lpips") is not None for d in sorted_res)
    if has_lpips:
        lpips_vals = [d["lpips"] if d.get("lpips") is not None else 0.0 for d in sorted_res]

    fig, axes = plt.subplots(1, 3 if has_lpips else 2, figsize=(16 if has_lpips else 11, 4.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    # 1. Utility: Accuracy vs Sigma
    ax1 = axes[0]
    ax1.plot(sigmas, accs, marker="o", color="#1f77b4", linewidth=2.2, markersize=8, label="Test Accuracy (%)")
    for s, a in zip(sigmas, accs):
        ax1.annotate(f"{a:.2f}%", (s, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color="#1f77b4")
    ax1.set_title("Utility: Phân loại CIFAR-10", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel(r"Mức nhiễu Gaussian ($\sigma$)", fontsize=11)
    ax1.set_ylabel("Độ chính xác Test (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xticks(sigmas)

    # 2. Security: PSNR & SSIM vs Sigma
    ax2 = axes[1]
    color_psnr = "#d62728"
    color_ssim = "#2ca02c"
    
    line1 = ax2.plot(sigmas, psnrs, marker="s", color=color_psnr, linewidth=2.0, markersize=7, label="PSNR (dB)")
    ax2.set_xlabel(r"Mức nhiễu Gaussian ($\sigma$)", fontsize=11)
    ax2.set_ylabel("PSNR (dB)", color=color_psnr, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_psnr)
    ax2.set_xticks(sigmas)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(sigmas, ssims, marker="^", color=color_ssim, linewidth=2.0, markersize=7, linestyle="--", label="SSIM")
    ax2_twin.set_ylabel("SSIM", color=color_ssim, fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor=color_ssim)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right", framealpha=0.9)
    ax2.set_title("Security: Khả năng Tái tạo (PSNR & SSIM)", fontsize=12, fontweight="bold", pad=10)

    # 3. LPIPS vs Sigma (nếu có)
    if has_lpips:
        ax3 = axes[2]
        ax3.plot(sigmas, lpips_vals, marker="D", color="#9467bd", linewidth=2.0, markersize=7, label="LPIPS Distance")
        for s, lp in zip(sigmas, lpips_vals):
            ax3.annotate(f"{lp:.3f}", (s, lp), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#9467bd")
        ax3.set_title("Perceptual Distance (LPIPS)", fontsize=12, fontweight="bold", pad=10)
        ax3.set_xlabel(r"Mức nhiễu Gaussian ($\sigma$)", fontsize=11)
        ax3.set_ylabel("LPIPS (Càng cao càng méo)", fontsize=11)
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.set_xticks(sigmas)

    plt.suptitle("ĐÁNH ĐỔI PRIVACY - UTILITY VỚI PHÒNG THỦ NHIỄU GAUSSIAN (B1)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[PLOT] Đã lưu đồ thị Trade-off tại: {save_path}", flush=True)

    if show:
        plt.show()
    plt.close()


def plot_reconstruction_grid(originals, recons_by_sigma, mean, std, save_path=None, num_images=6, show=False):
    """
    Xuất lưới ảnh so sánh:
    Hàng 1: Ảnh gốc
    Hàng 2..N: Ảnh tái tạo tại từng mức sigma
    """
    sigmas = sorted(list(recons_by_sigma.keys()))
    num_rows = 1 + len(sigmas)
    n_imgs = min(num_images, originals.size(0))

    fig, axes = plt.subplots(num_rows, n_imgs, figsize=(2.2 * n_imgs, 2.3 * num_rows), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    orig_denorm = denormalize(originals[:n_imgs], mean, std).detach().cpu().numpy()

    # Hàng 1: Ảnh gốc
    for j in range(n_imgs):
        ax = axes[0, j] if num_rows > 1 else axes[j]
        img = np.transpose(orig_denorm[j], (1, 2, 0))
        ax.imshow(img)
        ax.axis("off")
        if j == 0:
            ax.set_title("Ảnh Gốc", fontsize=11, fontweight="bold", loc="left")

    # Các hàng tiếp theo: Tái tạo tại từng sigma
    for row_idx, sigma in enumerate(sigmas, start=1):
        recon_tensor = recons_by_sigma[sigma][:n_imgs]
        recon_denorm = denormalize(recon_tensor, mean, std).detach().cpu().numpy()
        for j in range(n_imgs):
            ax = axes[row_idx, j]
            img = np.transpose(recon_denorm[j], (1, 2, 0))
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_title(f"Tái tạo (σ={sigma})", fontsize=11, fontweight="bold", loc="left")

    plt.suptitle("SO SÁNH CHẤT LƯỢNG TÁI TẠO ẢNH THEO MỨC NHIỄU GAUSSIAN (B1)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[PLOT] Đã lưu lưới ảnh tái tạo tại: {save_path}", flush=True)

    if show:
        plt.show()
    plt.close()
