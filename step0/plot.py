# Bước 0 — Module vẽ đồ thị trực quan hóa quá trình huấn luyện
import os
import sys
import json
import argparse
import matplotlib
# Thiết lập backend không phụ thuộc GUI khi chạy trên server/terminal
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history, save_path="training_curves.png", show=False):
    """
    Vẽ các biểu đồ trực quan hóa quá trình huấn luyện:
    - Đồ thị 1: Train Loss & Test Loss
    - Đồ thị 2: Train Accuracy & Test Accuracy (%)
    - Đồ thị 3: Learning Rate
    - Đồ thị 4: Thời gian mỗi Epoch (giây)

    history: Danh sách các dict với các keys:
             ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'lr', 'epoch_time']
    """
    if not history:
        print("[WARN] Lịch sử huấn luyện rỗng, bỏ qua vẽ đồ thị.")
        return

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    train_accs = [item.get("train_acc", 0.0) * 100 for item in history]
    lrs = [item.get("lr", 0.0) for item in history]
    times = [item.get("epoch_time", 0.0) for item in history]

    # Lọc các điểm có đánh giá test (có thể eval định kỳ không phải epoch nào cũng có)
    eval_epochs = [item["epoch"] for item in history if item.get("test_acc") is not None]
    test_accs = [item["test_acc"] * 100 for item in history if item.get("test_acc") is not None]
    test_losses = [item["test_loss"] for item in history if item.get("test_loss") is not None]

    plt.figure(figsize=(14, 10))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Biểu đồ Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    if test_losses and len(test_losses) == len(eval_epochs):
        plt.plot(eval_epochs, test_losses, label="Test Loss", color="#d62728", linestyle="--", marker="o", markersize=4, linewidth=1.5)
    plt.title("Hàm mất mát (Loss)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 2. Biểu đồ Accuracy
    plt.subplot(2, 2, 2)
    if any(train_accs):
        plt.plot(epochs, train_accs, label="Train Acc (%)", color="#2ca02c", linewidth=2)
    if test_accs:
        plt.plot(eval_epochs, test_accs, label="Test Acc (%)", color="#ff7f0e", linestyle="--", marker="s", markersize=4, linewidth=1.5)
        # Đánh dấu điểm Test Acc cao nhất
        best_idx = max(range(len(test_accs)), key=lambda i: test_accs[i])
        best_ep = eval_epochs[best_idx]
        best_val = test_accs[best_idx]
        plt.scatter([best_ep], [best_val], color="red", s=80, zorder=5, label=f"Best: {best_val:.2f}% (ep {best_ep})")
        plt.annotate(f"{best_val:.2f}%", (best_ep, best_val), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontweight="bold", color="darkred")
    plt.title("Độ chính xác (Accuracy)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 3. Biểu đồ Learning Rate
    plt.subplot(2, 2, 3)
    plt.plot(epochs, lrs, label="Learning Rate", color="#9467bd", linewidth=2)
    plt.yscale("log")
    plt.title("Tốc độ học (Learning Rate)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("LR (log scale)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 4. Biểu đồ Thời gian mỗi Epoch
    plt.subplot(2, 2, 4)
    plt.bar(epochs, times, color="#17becf", alpha=0.7, width=0.8, label="Train Time / epoch")
    if times:
        avg_time = sum(times) / len(times)
        plt.axhline(avg_time, color="navy", linestyle=":", label=f"Avg: {avg_time:.1f}s")
    plt.title("Thời gian huấn luyện mỗi Epoch", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Thời gian (giây)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Đã lưu đồ thị trực quan tại: {save_path}", flush=True)

    if show:
        plt.show()
    plt.close()


def load_history_from_file(file_path):
    """Nạp lịch sử từ file .json hoặc checkpoint .pt"""
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif file_path.endswith(".pt"):
        import torch
        ckpt = torch.load(file_path, map_location="cpu")
        if "history" in ckpt:
            return ckpt["history"]
        else:
            raise ValueError(f"File checkpoint {file_path} không chứa key 'history'!")
    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {file_path}. Vui lòng dùng .json hoặc .pt")


def main():
    parser = argparse.ArgumentParser(description="Vẽ đồ thị quá trình huấn luyện từ history.json hoặc checkpoint .pt")
    parser.add_argument("--history", type=str, default="history.json", help="Đường dẫn file history.json hoặc checkpoint .pt")
    parser.add_argument("--output", type=str, default="training_curves.png", help="Đường dẫn lưu ảnh đồ thị")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ đồ thị")
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"[ERROR] Không tìm thấy file lịch sử: {args.history}")
        sys.exit(1)

    history = load_history_from_file(args.history)
    plot_history(history, save_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
