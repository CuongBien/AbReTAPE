# Bước 3 — Trực quan hóa đường cong Trade-off (DP-SGD: Epsilon / Sigma vs Utility & Security) và lưới ảnh tái tạo
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


def plot_b2_tradeoff(results, save_path=None, show=False):
    """
    Vẽ đồ thị Privacy-Utility Trade-off cho Baseline B2 (DP-SGD):
    - Subplot 1: sigma_DP vs Test Accuracy & PSNR/SSIM
    - Subplot 2: Epsilon (Ngân sách riêng tư DP) vs Test Accuracy & PSNR
    - Subplot 3 (nếu có): LPIPS vs sigma_DP
    results: List các dict, mỗi dict chứa:
      {"sigma_dp": float, "epsilon": float, "test_acc": float, "psnr": float, "ssim": float, "lpips": float}
    """
    if not results:
        return

    # Sắp xếp theo sigma_dp tăng dần
    sorted_res = sorted(results, key=lambda d: d.get("sigma_dp", 0.0))
    sigmas = [d.get("sigma_dp", 0.0) for d in sorted_res]
    epsilons = [d.get("epsilon", 0.0) for d in sorted_res]
    accs = [d["test_acc"] * 100.0 for d in sorted_res]
    psnrs = [d["psnr"] for d in sorted_res]
    ssims = [d["ssim"] for d in sorted_res]
    has_lpips = any(d.get("lpips") is not None for d in sorted_res)
    if has_lpips:
        lpips_vals = [d["lpips"] if d.get("lpips") is not None else 0.0 for d in sorted_res]

    num_cols = 3 if has_lpips else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(16 if has_lpips else 12, 5.0), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    # 1. Subplot 1: sigma_DP vs Utility & Security
    ax1 = axes[0]
    color_acc = "#1f77b4"
    color_psnr = "#d62728"

    line1 = ax1.plot(sigmas, accs, marker="o", color=color_acc, linewidth=2.2, markersize=8, label="Test Accuracy (%)")
    for s, a in zip(sigmas, accs):
        ax1.annotate(f"{a:.1f}%", (s, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold", color=color_acc)
    ax1.set_xlabel(r"Hệ số nhiễu Gradient ($\sigma_{\mathrm{DP}}$)", fontsize=11)
    ax1.set_ylabel("Độ chính xác Test (%)", color=color_acc, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticks(sigmas)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(sigmas, psnrs, marker="s", color=color_psnr, linewidth=2.0, markersize=7, linestyle="--", label="PSNR (dB)")
    for s, p in zip(sigmas, psnrs):
        ax1_twin.annotate(f"{p:.1f}dB", (s, p), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=color_psnr)
    ax1_twin.set_ylabel("PSNR Tái tạo (dB)", color=color_psnr, fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor=color_psnr)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", framealpha=0.9)
    ax1.set_title(r"Đánh đổi theo $\sigma_{\mathrm{DP}}$", fontsize=12, fontweight="bold", pad=10)

    # 2. Subplot 2: Epsilon (Privacy Budget) vs Utility & Security
    ax2 = axes[1]
    sorted_by_eps = sorted(zip(epsilons, accs, psnrs, ssims), key=lambda t: t[0])
    eps_s, acc_s, psnr_s, ssim_s = zip(*sorted_by_eps)

    color_eps_acc = "#2ca02c"
    color_eps_ssim = "#ff7f0e"

    l1 = ax2.plot(eps_s, acc_s, marker="^", color=color_eps_acc, linewidth=2.2, markersize=8, label="Test Accuracy (%)")
    for e, a in zip(eps_s, acc_s):
        ax2.annotate(f"{a:.1f}%", (e, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold", color=color_eps_acc)
    ax2.set_xlabel(r"Ngân sách riêng tư ($\epsilon$ tại $\delta=10^{-5}$)", fontsize=11)
    ax2.set_ylabel("Độ chính xác Test (%)", color=color_eps_acc, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_eps_acc)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2_twin = ax2.twinx()
    l2 = ax2_twin.plot(eps_s, ssim_s, marker="D", color=color_eps_ssim, linewidth=2.0, markersize=7, linestyle="--", label="SSIM Tái tạo")
    for e, ss in zip(eps_s, ssim_s):
        ax2_twin.annotate(f"{ss:.3f}", (e, ss), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=color_eps_ssim)
    ax2_twin.set_ylabel("SSIM Tái tạo", color=color_eps_ssim, fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor=color_eps_ssim)

    lines2 = l1 + l2
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc="lower right", framealpha=0.9)
    ax2.set_title(r"Đánh đổi theo Ngân sách $\epsilon$ (RDP)", fontsize=12, fontweight="bold", pad=10)

    # 3. Subplot 3 (nếu có LPIPS): LPIPS vs sigma_DP
    if has_lpips:
        ax3 = axes[2]
        ax3.plot(sigmas, lpips_vals, marker="X", color="#9467bd", linewidth=2.0, markersize=8, label="LPIPS Distance")
        for s, lp in zip(sigmas, lpips_vals):
            ax3.annotate(f"{lp:.3f}", (s, lp), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#9467bd")
        ax3.set_title("Perceptual Distance (LPIPS)", fontsize=12, fontweight="bold", pad=10)
        ax3.set_xlabel(r"Hệ số nhiễu Gradient ($\sigma_{\mathrm{DP}}$)", fontsize=11)
        ax3.set_ylabel("LPIPS (Càng cao càng méo)", fontsize=11)
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.set_xticks(sigmas)

    plt.suptitle("ĐÁNH ĐỔI PRIVACY - UTILITY VỚI PHÒNG THỦ DP-SGD TRÊN CLIENT (B2)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[PLOT] Đã lưu đồ thị Trade-off B2 tại: {save_path}", flush=True)

    if show:
        plt.show()
    plt.close()


def plot_b2_reconstruction_grid(originals, recons_by_sigma, eps_by_sigma=None, mean=None, std=None, save_path=None, num_images=6, show=False):
    """
    Xuất lưới ảnh so sánh tái tạo:
    Hàng 1: Ảnh gốc CIFAR-10
    Hàng 2..N: Ảnh tái tạo từ IR của Client được train với từng mức sigma_DP (kèm epsilon)
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

    # Các hàng tiếp theo: Tái tạo tại từng mức sigma_DP
    for row_idx, sigma in enumerate(sigmas, start=1):
        recon_tensor = recons_by_sigma[sigma][:n_imgs]
        recon_denorm = denormalize(recon_tensor, mean, std).detach().cpu().numpy()
        eps_info = f", ε={eps_by_sigma[sigma]:.2f}" if eps_by_sigma and sigma in eps_by_sigma else ""
        for j in range(n_imgs):
            ax = axes[row_idx, j]
            img = np.transpose(recon_denorm[j], (1, 2, 0))
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_title(f"Tái tạo (σ={sigma}{eps_info})", fontsize=10, fontweight="bold", loc="left")

    plt.suptitle("SO SÁNH TÁI TẠO ẢNH TRÊN CLIENT HUẤN LUYỆN BẰNG DP-SGD (B2)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[PLOT] Đã lưu lưới ảnh tái tạo B2 tại: {save_path}", flush=True)

    if show:
        plt.show()
    plt.close()
