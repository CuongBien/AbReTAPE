# Bước 1 — Module trực quan hóa: xuất ảnh đối chứng và đồ thị tấn công
import os
import sys
import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from metrics import denormalize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


@torch.no_grad()
def save_reconstruction_grid(client, decoder, loader, device, mean, std,
                             save_path="reconstruction_grid.png", num_images=8):
    """
    Xuất ảnh lưới trực quan so sánh ảnh gốc (Original) và ảnh tái tạo (Reconstructed).
    - Hàng trên: Ảnh gốc x
    - Hàng dưới: Ảnh tái tạo x_hat kèm PSNR/SSIM từng ảnh
    """
    client.eval()
    decoder.eval()

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
    x_hat = decoder(z)

    x_orig = denormalize(x_all, mean, std).cpu().numpy()
    x_rec = denormalize(x_hat, mean, std).cpu().numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2.2, 5.0))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(num_images):
        img_orig = x_orig[i].transpose(1, 2, 0)
        img_rec = x_rec[i].transpose(1, 2, 0)

        # Tính PSNR và SSIM từng cặp
        p = peak_signal_noise_ratio(img_orig, img_rec, data_range=1.0)
        s = structural_similarity(img_orig, img_rec, channel_axis=2, data_range=1.0)
        class_name = CIFAR10_CLASSES[y_all[i]] if y_all[i] < len(CIFAR10_CLASSES) else f"C{y_all[i]}"

        # Hàng 1: Ảnh gốc
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(f"Gốc: {class_name}", fontsize=10, fontweight="bold")
        axes[0, i].axis("off")

        # Hàng 2: Ảnh tái tạo
        axes[1, i].imshow(img_rec)
        axes[1, i].set_title(f"PSNR: {p:.1f}dB\nSSIM: {s:.2f}", fontsize=9, color="darkgreen" if p > 20 else "darkred")
        axes[1, i].axis("off")

    fig.suptitle("So sánh Trực quan: Ảnh gốc (Hàng 1) vs Ảnh Decoder tái tạo (Hàng 2)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Đã xuất lưới ảnh đối chứng tại: {save_path}", flush=True)


def plot_attack_curves(history, save_path="attack_curves.png", show=False):
    """
    Vẽ đồ thị theo dõi quá trình tấn công:
    - 1. MSE Loss (Train vs Test)
    - 2. PSNR (dB) kèm mốc tiêu chuẩn 20 dB
    - 3. SSIM kèm mốc tiêu chuẩn 0.6
    - 4. LPIPS hoặc Thời gian train
    """
    if not history:
        print("[WARN] Lịch sử tấn công rỗng, bỏ qua vẽ đồ thị.")
        return

    epochs = [item["epoch"] for item in history]
    train_mses = [item["train_mse"] for item in history]
    eval_epochs = [item["epoch"] for item in history if item.get("test_psnr") is not None]
    test_mses = [item["test_mse"] for item in history if item.get("test_mse") is not None]
    test_psnrs = [item["test_psnr"] for item in history if item.get("test_psnr") is not None]
    test_ssims = [item["test_ssim"] for item in history if item.get("test_ssim") is not None]
    test_lpips = [item.get("test_lpips") for item in history if item.get("test_lpips") is not None]

    plt.figure(figsize=(14, 10))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Biểu đồ MSE Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_mses, label="Train MSE Loss", color="#1f77b4", linewidth=2)
    if test_mses:
        plt.plot(eval_epochs, test_mses, label="Test MSE Loss", color="#d62728", linestyle="--", marker="o", markersize=4)
    plt.title("MSE Loss Tái tạo qua các Epoch", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("MSE", fontsize=11)
    plt.yscale("log")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 2. Biểu đồ PSNR (dB)
    plt.subplot(2, 2, 2)
    if test_psnrs:
        plt.plot(eval_epochs, test_psnrs, label="Test PSNR (dB)", color="#2ca02c", linewidth=2, marker="s", markersize=4)
        best_p_idx = max(range(len(test_psnrs)), key=lambda i: test_psnrs[i])
        best_p_val = test_psnrs[best_p_idx]
        best_p_ep = eval_epochs[best_p_idx]
        plt.scatter([best_p_ep], [best_p_val], color="red", s=70, zorder=5, label=f"Max PSNR: {best_p_val:.2f} dB")
        plt.annotate(f"{best_p_val:.1f} dB", (best_p_ep, best_p_val), textcoords="offset points", xytext=(0, 7), ha="center", fontweight="bold", color="darkred")
    # Đường ngưỡng chấp nhận 20 dB
    plt.axhline(20.0, color="crimson", linestyle=":", linewidth=1.8, label="Ngưỡng đạt (20 dB)")
    plt.title("Chỉ số PSNR (Peak Signal-to-Noise Ratio)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("PSNR (dB)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 3. Biểu đồ SSIM
    plt.subplot(2, 2, 3)
    if test_ssims:
        plt.plot(eval_epochs, test_ssims, label="Test SSIM", color="#ff7f0e", linewidth=2, marker="^", markersize=4)
        best_s_idx = max(range(len(test_ssims)), key=lambda i: test_ssims[i])
        best_s_val = test_ssims[best_s_idx]
        best_s_ep = eval_epochs[best_s_idx]
        plt.scatter([best_s_ep], [best_s_val], color="red", s=70, zorder=5, label=f"Max SSIM: {best_s_val:.4f}")
        plt.annotate(f"{best_s_val:.3f}", (best_s_ep, best_s_val), textcoords="offset points", xytext=(0, 7), ha="center", fontweight="bold", color="darkred")
    # Đường ngưỡng chấp nhận 0.6
    plt.axhline(0.60, color="crimson", linestyle=":", linewidth=1.8, label="Ngưỡng đạt (0.60)")
    plt.title("Chỉ số SSIM (Structural Similarity)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("SSIM", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 4. Biểu đồ LPIPS hoặc Thời gian
    plt.subplot(2, 2, 4)
    if test_lpips and len(test_lpips) == len(eval_epochs):
        plt.plot(eval_epochs, test_lpips, label="Test LPIPS (càng thấp càng tốt)", color="#9467bd", linewidth=2, marker="d", markersize=4)
        plt.title("Chỉ số LPIPS (Perceptual Distance)", fontsize=12, fontweight="bold")
        plt.ylabel("LPIPS", fontsize=11)
    else:
        times = [item.get("epoch_time", 0.0) for item in history]
        plt.bar(epochs, times, color="#17becf", alpha=0.7, label="Thời gian train (s)")
        plt.title("Thời gian huấn luyện mỗi Epoch", fontsize=12, fontweight="bold")
        plt.ylabel("Thời gian (giây)", fontsize=11)
    plt.xlabel("Epoch", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Đã lưu đồ thị tấn công tại: {save_path}", flush=True)

    if show:
        plt.show()
    plt.close()
