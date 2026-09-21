# Module trực quan hóa: biểu đồ huấn luyện, tấn công, heatmap hấp thụ và lưới ảnh tái tạo
import os
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from ..metrics.reconstruction import denormalize

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def plot_training_curves(history, save_path="training_curves.png"):
    """
    Vẽ các biểu đồ trực quan hóa quá trình huấn luyện:
    - Đồ thị 1: Train Loss & Test Loss
    - Đồ thị 2: Train Accuracy & Test Accuracy (%)
    - Đồ thị 3: Learning Rate
    - Đồ thị 4: Thời gian mỗi Epoch (giây)
    """
    if not history:
        print("[WARN] Lịch sử huấn luyện rỗng, bỏ qua vẽ đồ thị.")
        return

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    train_accs = [item.get("train_acc", 0.0) * 100 for item in history]
    lrs = [item.get("lr", 0.0) for item in history]
    times = [item.get("epoch_time", 0.0) for item in history]

    eval_epochs = [item["epoch"] for item in history if item.get("test_acc") is not None]
    test_accs = [item["test_acc"] * 100 for item in history if item.get("test_acc") is not None]
    test_losses = [item["test_loss"] for item in history if item.get("test_loss") is not None]

    plt.figure(figsize=(14, 10))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    if test_losses and len(test_losses) == len(eval_epochs):
        plt.plot(eval_epochs, test_losses, label="Test Loss", color="#d62728", linestyle="--", marker="o", markersize=4, linewidth=1.5)
    plt.title("Hàm mất mát (Loss)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 2. Accuracy
    plt.subplot(2, 2, 2)
    if any(train_accs):
        plt.plot(epochs, train_accs, label="Train Acc (%)", color="#2ca02c", linewidth=2)
    if test_accs:
        plt.plot(eval_epochs, test_accs, label="Test Acc (%)", color="#ff7f0e", linestyle="--", marker="s", markersize=4, linewidth=1.5)
        best_idx = max(range(len(test_accs)), key=lambda i: test_accs[i])
        best_ep = eval_epochs[best_idx]
        best_ac = test_accs[best_idx]
        plt.scatter([best_ep], [best_ac], color="#d62728", s=100, zorder=5, label=f"Best: {best_ac:.2f}% (Ep {best_ep})")
    plt.title("Độ chính xác (Accuracy)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 3. Learning Rate
    plt.subplot(2, 2, 3)
    plt.plot(epochs, lrs, label="Learning Rate", color="#9467bd", linewidth=2)
    plt.title("Tốc độ học (Learning Rate Schedule)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Learning Rate", fontsize=11)
    plt.yscale("log" if max(lrs) / (min(lrs) + 1e-8) > 20 else "linear")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 4. Thời gian Epoch
    plt.subplot(2, 2, 4)
    plt.bar(epochs, times, color="#17becf", alpha=0.7, width=0.8, label="Thời gian (s)")
    avg_time = sum(times) / len(times) if times else 0.0
    plt.axhline(avg_time, color="#d62728", linestyle="--", label=f"Trung bình: {avg_time:.1f}s")
    plt.title("Thời gian Huấn luyện mỗi Epoch", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Thời gian (giây)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Đã lưu đồ thị huấn luyện tại: {save_path}")


def plot_attack_curves(history, save_path="attack_curves.png"):
    """
    Vẽ 4 đồ thị đánh giá quá trình huấn luyện bộ giải mã tấn công Feature Inversion:
    1. MSE Loss (Train vs Test)
    2. PSNR (dB)
    3. SSIM
    4. LPIPS
    """
    if not history:
        print("[WARN] Lịch sử tấn công rỗng, bỏ qua vẽ đồ thị.")
        return

    epochs = [item["epoch"] for item in history]
    train_mse = [item.get("train_mse", 0.0) for item in history]
    eval_epochs = [item["epoch"] for item in history if item.get("test_mse") is not None]
    test_mse = [item["test_mse"] for item in history if item.get("test_mse") is not None]
    test_psnr = [item["test_psnr"] for item in history if item.get("test_psnr") is not None]
    test_ssim = [item["test_ssim"] for item in history if item.get("test_ssim") is not None]
    test_lpips = [item["test_lpips"] for item in history if item.get("test_lpips") is not None]

    plt.figure(figsize=(14, 10))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. MSE
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_mse, label="Train MSE", color="#1f77b4", linewidth=2)
    if test_mse:
        plt.plot(eval_epochs, test_mse, label="Test MSE", color="#d62728", linestyle="--", marker="o", markersize=4)
    plt.title("Hàm mất mát Tái tạo (MSE)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # 2. PSNR
    plt.subplot(2, 2, 2)
    if test_psnr:
        plt.plot(eval_epochs, test_psnr, label="Test PSNR", color="#2ca02c", marker="s", markersize=4, linewidth=2)
        best_psnr = max(test_psnr)
        plt.axhline(best_psnr, color="#d62728", linestyle=":", label=f"Max PSNR: {best_psnr:.2f} dB")
    plt.title("Tỷ số Tín hiệu trên Nhiễu đỉnh (PSNR - dB)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # 3. SSIM
    plt.subplot(2, 2, 3)
    if test_ssim:
        plt.plot(eval_epochs, test_ssim, label="Test SSIM", color="#ff7f0e", marker="^", markersize=4, linewidth=2)
        best_ssim = max(test_ssim)
        plt.axhline(best_ssim, color="#d62728", linestyle=":", label=f"Max SSIM: {best_ssim:.4f}")
    plt.title("Độ tương đồng Cấu trúc (SSIM)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM (0 -> 1)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # 4. LPIPS
    plt.subplot(2, 2, 4)
    if test_lpips:
        plt.plot(eval_epochs, test_lpips, label="Test LPIPS", color="#9467bd", marker="d", markersize=4, linewidth=2)
        best_lpips = min(test_lpips)
        plt.axhline(best_lpips, color="#d62728", linestyle=":", label=f"Min LPIPS: {best_lpips:.4f}")
    plt.title("Khoảng cách Tri giác (LPIPS - Càng thấp càng giống)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("LPIPS")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Đã lưu đồ thị tấn công tại: {save_path}")


@torch.no_grad()
def save_reconstruction_grid(client, decoder, loader, device, mean, std,
                             save_path="reconstruction_grid.png", num_images=8, defense=None):
    """
    Xuất ảnh lưới so sánh ảnh gốc và ảnh tái tạo kèm PSNR/SSIM từng ảnh.
    """
    client.eval()
    decoder.eval()
    if defense is not None:
        defense.eval()

    images_collected = []
    targets_collected = []

    for x, y in loader:
        images_collected.append(x)
        targets_collected.append(y)
        if sum(t.size(0) for t in targets_collected) >= num_images:
            break

    x_all = torch.cat(images_collected, dim=0)[:num_images].to(device)
    y_all = torch.cat(targets_collected, dim=0)[:num_images].cpu().numpy()

    z = client(x_all)
    if defense is not None:
        z = defense(z)
    x_hat = decoder(z)

    x_orig = denormalize(x_all, mean, std).cpu().numpy()
    x_rec = denormalize(x_hat, mean, std).cpu().numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2.2, 5.0))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(num_images):
        img_orig = x_orig[i].transpose(1, 2, 0)
        img_rec = x_rec[i].transpose(1, 2, 0)

        p = peak_signal_noise_ratio(img_orig, img_rec, data_range=1.0)
        s = structural_similarity(img_orig, img_rec, channel_axis=2, data_range=1.0)
        class_name = CIFAR10_CLASSES[y_all[i]] if y_all[i] < len(CIFAR10_CLASSES) else f"C{y_all[i]}"

        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(f"Gốc: {class_name}", fontsize=10, fontweight="bold")
        axes[0, i].axis("off")

        axes[1, i].imshow(img_rec)
        axes[1, i].set_title(f"{p:.1f}dB | {s:.2f}", fontsize=9, color="#d62728" if p < 20 else "#2ca02c")
        axes[1, i].axis("off")

    axes[0, 0].text(-0.25, 0.5, "Ảnh Gốc\nx", transform=axes[0, 0].transAxes,
                     fontsize=12, fontweight="bold", va="center", ha="right")
    axes[1, 0].text(-0.25, 0.5, "Tái Tạo\n\\hat{x}", transform=axes[1, 0].transAxes,
                     fontsize=12, fontweight="bold", va="center", ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Đã xuất lưới ảnh đối chứng tại: {save_path}")


def plot_permutation_matrix(matrix, W_ref=None, perm_true=None, save_path="permutation_heatmap.png", title="Ma trận Trọng số Hấp thụ"):
    """
    Vẽ Heatmap trực quan hóa ma trận hoán vị hoặc ma trận học được của Adapter.
    Hỗ trợ cả vẽ đơn (Adapter weights) lẫn vẽ đôi đối chứng (W_hat vs W_ref theo perm_true).
    """
    if W_ref is not None and perm_true is not None:
        from ..attacks.recover_perm import compute_cosine_similarity_matrix
        if torch.is_tensor(perm_true):
            perm_true = perm_true.cpu().numpy()
        elif isinstance(perm_true, list):
            perm_true = np.array(perm_true)

        sim_matrix = compute_cosine_similarity_matrix(matrix, W_ref)
        num_channels = sim_matrix.shape[0]
        sim_aligned = np.zeros_like(sim_matrix)
        for c in range(num_channels):
            target_channel = perm_true[c]
            sim_aligned[target_channel, :] = sim_matrix[c, :]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        im1 = axes[0].imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=-0.2, vmax=1.0)
        axes[0].set_title("(a) Tương quan Chưa sắp xếp\n(Thứ tự kênh quan sát tại Server)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Chỉ số kênh Server Tham chiếu (W_ref)", fontsize=11)
        axes[0].set_ylabel("Chỉ số kênh Server Huấn luyện (W)", fontsize=11)
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(sim_aligned, cmap="plasma", aspect="auto", vmin=-0.2, vmax=1.0)
        axes[1].set_title("(b) Đã sắp xếp lại theo Hoán vị Thật $\\pi$\n(Đường chéo sáng = Hấp thụ thành công)", fontsize=12, fontweight="bold", color="darkred")
        axes[1].set_xlabel("Chỉ số kênh Server Tham chiếu (W_ref)", fontsize=11)
        axes[1].set_ylabel("Chỉ số kênh sau Hoán vị $\\pi$", fontsize=11)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
    else:
        if torch.is_tensor(matrix):
            matrix = matrix.detach().cpu().numpy()

        plt.figure(figsize=(9, 8))
        plt.imshow(np.abs(matrix), cmap="viridis", aspect="auto")
        plt.colorbar(label="Trọng số tuyệt đối |A[r, c]|")
        plt.title(title, fontsize=13, fontweight="bold")
        plt.xlabel("Kênh vào c (z_permuted)")
        plt.ylabel("Kênh ra r (z_hat)")
        plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Đã lưu heatmap ma trận tại: {save_path}")


def plot_tradeoff_curves(results, save_path="tradeoff_curves.png", x_key="sigma", x_label="Cường độ Nhiễu (sigma)"):
    """
    Vẽ đồ thị đánh đổi Utility (Accuracy) vs Privacy (PSNR, SSIM, LPIPS).
    """
    if not results:
        return

    x_vals = [r[x_key] for r in results]
    accs = [r["test_acc"] * 100 for r in results]
    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    color = "#1f77b4"
    ax1.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)", color=color, fontsize=12, fontweight="bold")
    l1 = ax1.plot(x_vals, accs, color=color, marker="o", linewidth=2.5, label="Test Acc (%)")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("PSNR tái tạo (dB - càng thấp càng an toàn)", color=color2, fontsize=12, fontweight="bold")
    l2 = ax2.plot(x_vals, psnrs, color=color2, marker="s", linestyle="--", linewidth=2.5, label="Recon PSNR (dB)")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)
    plt.title(f"Đánh đổi Tiện ích - Riêng tư (Utility vs Privacy Trade-off)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Đã lưu đồ thị Trade-off tại: {save_path}")


# Aliases
plot_history = plot_training_curves
plot_step2_curves = plot_training_curves
plot_step1_curves = plot_attack_curves
