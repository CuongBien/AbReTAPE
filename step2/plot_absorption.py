# Bước 2 — Module trực quan hóa ma trận hấp thụ hoán vị và đồ thị huấn luyện
import os
import sys
import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from recover import compute_cosine_similarity_matrix, compute_cost_matrix


def plot_permutation_matrix(W, W_ref, perm_true, save_path="permutation_heatmap.png", show=False):
    """
    Vẽ 2 Heatmap ma trận so sánh tương quan 64x64 giữa Server W và Server tham chiếu W_ref:
    1. Trái: Trước khi sắp xếp (Ma trận hỗn loạn do hoán vị kênh).
    2. Phải: Sau khi sắp xếp lại theo hoán vị pi (Hiện rõ đường chéo chính,
       chứng minh Server đã tự động học giải mã hoán vị).
    """
    if torch.is_tensor(perm_true):
        perm_true = perm_true.cpu().numpy()
    elif isinstance(perm_true, list):
        perm_true = np.array(perm_true)

    # Tính ma trận tương đồng cosine [64, 64]
    sim_matrix = compute_cosine_similarity_matrix(W, W_ref)
    num_channels = sim_matrix.shape[0]

    # Ma trận đã sắp xếp lại theo hoán vị thật pi:
    # Kênh c của W tương ứng với kênh perm_true[c] của W_ref.
    # Khi sắp xếp các hàng theo perm_true, phần tử đường chéo sẽ là khớp chính xác.
    sim_aligned = np.zeros_like(sim_matrix)
    for c in range(num_channels):
        target_channel = perm_true[c]
        sim_aligned[target_channel, :] = sim_matrix[c, :]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # 1. Ma trận trước khi sắp xếp
    im1 = axes[0].imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=-0.2, vmax=1.0)
    axes[0].set_title("(a) Tương quan Chưa sắp xếp\n(Thứ tự kênh quan sát tại Server)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Chỉ số kênh Server Tham chiếu (W_ref)", fontsize=11)
    axes[0].set_ylabel("Chỉ số kênh Server Huấn luyện (W)", fontsize=11)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Ma trận sau khi sắp xếp lại theo pi
    im2 = axes[1].imshow(sim_aligned, cmap="plasma", aspect="auto", vmin=-0.2, vmax=1.0)
    axes[1].set_title("(b) Đã sắp xếp lại theo Hoán vị Thật $\\pi$\n(Đường chéo sáng rực rỡ = Hấp thụ thành công)", fontsize=12, fontweight="bold", color="darkred")
    axes[1].set_xlabel("Chỉ số kênh Server Tham chiếu (W_ref)", fontsize=11)
    axes[1].set_ylabel("Chỉ số kênh sau Hoán vị $\\pi$", fontsize=11)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Kiểm chứng Thực nghiệm Hấp thụ (Absorption Verification): Server tự học ra $E^{-1}$", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Đã xuất Heatmap ma trận hấp thụ tại: {save_path}", flush=True)

    if show:
        plt.show()
    plt.close()


def plot_step2_curves(history, save_path="step2_curves.png", show=False):
    """
    Vẽ đồ thị theo dõi quá trình huấn luyện Bước 2:
    - Đồ thị 1: Train Loss & Test Loss
    - Đồ thị 2: Test Accuracy (%)
    - Đồ thị 3: Tỷ lệ khớp hoán vị (%) kèm mốc 80% và Random Baseline (1.56%)
    - Đồ thị 4: Learning Rate / Thời gian mỗi epoch
    """
    if not history:
        print("[WARN] Lịch sử Bước 2 rỗng, bỏ qua vẽ đồ thị.")
        return

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    eval_epochs = [item["epoch"] for item in history if item.get("test_acc") is not None]
    test_losses = [item["test_loss"] for item in history if item.get("test_loss") is not None]
    test_accs = [item["test_acc"] * 100 for item in history if item.get("test_acc") is not None]

    match_epochs = [item["epoch"] for item in history if item.get("match_acc") is not None]
    match_accs = [item["match_acc"] * 100 for item in history if item.get("match_acc") is not None]
    rand_accs = [item.get("rand_acc", 0.0156) * 100 for item in history if item.get("match_acc") is not None]

    plt.figure(figsize=(14, 10))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Loss Curve
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    if test_losses:
        plt.plot(eval_epochs, test_losses, label="Test Loss", color="#d62728", linestyle="--", marker="o", markersize=4)
    plt.title("Hàm mất mát (Cross-Entropy Loss)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 2. Accuracy Curve
    plt.subplot(2, 2, 2)
    if test_accs:
        plt.plot(eval_epochs, test_accs, label="Test Acc (%)", color="#2ca02c", linewidth=2, marker="s", markersize=4)
        best_acc = max(test_accs)
        plt.axhline(best_acc, color="darkgreen", linestyle=":", label=f"Best: {best_acc:.2f}%")
    plt.title("Độ chính xác phân loại (Classification Accuracy)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 3. Tỷ lệ khớp Hoán vị (%)
    plt.subplot(2, 2, 3)
    if match_accs:
        plt.plot(match_epochs, match_accs, label="Độ khớp Hoán vị khôi phục (%)", color="#9467bd", linewidth=2.5, marker="o", markersize=5)
        # Đường ngưỡng 80%
        plt.axhline(80.0, color="crimson", linestyle="--", linewidth=1.8, label="Ngưỡng đạt tiêu chuẩn (80%)")
        # Đường đối chứng ngẫu nhiên (~1.56%)
        plt.axhline(1.56, color="gray", linestyle=":", linewidth=1.5, label="Đối chứng ngẫu nhiên (1.56%)")
        best_m = max(match_accs)
        plt.scatter([match_epochs[match_accs.index(best_m)]], [best_m], color="red", s=70, zorder=5)
        plt.annotate(f"{best_m:.1f}%", (match_epochs[match_accs.index(best_m)], best_m),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="darkred")
    plt.title("Tỷ lệ khớp Hoán vị (Permutation Recovery Rate)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Tỷ lệ khớp (%)", fontsize=11)
    plt.legend(loc="center right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 4. Learning Rate & Time
    plt.subplot(2, 2, 4)
    lrs = [item.get("lr", 0.1) for item in history]
    plt.plot(epochs, lrs, label="Learning Rate", color="#e377c2", linewidth=2)
    plt.yscale("log")
    plt.title("Tốc độ học (Learning Rate Schedule)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("LR (log scale)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Đã lưu đồ thị huấn luyện Bước 2 tại: {save_path}", flush=True)

    if show:
        plt.show()
    plt.close()
