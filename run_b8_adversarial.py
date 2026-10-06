#!/usr/bin/env python3
# ==============================================================================
# BASELINE B8: ADVERSARIAL TRAINING (MIN-MAX PRIVACY DEFENSE)
# ==============================================================================
# - Testbed: CIFAR-10, Split Learning (Client: ResNet18 stem+layer1, Server: layer2..4+fc)
# - Vòng lặp xen kẽ (Alternating Min-Max) trong từng batch suốt 100 epochs:
#     * Bước Max: Adversary học giải mã z hiện tại (Adam lr 1e-3, 1 step)
#     * Bước Min: Client + Server tối ưu L = CE(S(z), y) - lam * clamp(rec, max=cap)
# - Single forward pass: Client forward đúng 1 lần mỗi batch để không nhân đôi thống kê BatchNorm.
# - Gradient isolation: Tạm thời tắt requires_grad của Adversary ở bước Min để chống tích lũy rò rỉ.
# - BatchNorm shift check: Đánh giá Adversary trên cả z_eval và z_train, lấy max SSIM khi tính G.
# - Quét lambda in {0.1, 0.5, 1.0, 5.0} với ngưỡng chọn Task Acc >= 92.7% (mất <= 2% vs B0 94.71%).
# ==============================================================================
import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel, ServerModel
from src.data.cifar import CIFAR10_MEAN, CIFAR10_STD, denormalize
from src.attacks import Decoder
from src.metrics.reconstruction import psnr_ssim, distance_correlation


class LearnedAdaptiveDecoderB8(nn.Module):
    """
    Biến thể B8-mlp: Adversary mạnh nhất (MLP Unprojector 65536 -> 512 -> 65536 + ConvDecoder).
    """
    def __init__(self, in_features=65536, D=65536, hidden_dim=512):
        super().__init__()
        self.unprojector = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D),
            nn.BatchNorm1d(D)
        )
        self.conv_decoder = Decoder(in_channels=64, out_channels=3)

    def forward(self, z):
        if z.dim() > 2:
            z_flat = z.view(z.size(0), -1)
        else:
            z_flat = z
        z_unproj = self.unprojector(z_flat).view(-1, 64, 32, 32)
        return self.conv_decoder(z_unproj)


def get_b8_data_loaders(data_dir="./data", batch_size=128, test_eval_size=2000, num_workers=0, seed=42):
    """
    Nạp dữ liệu chuẩn cho B8:
    - Trainloader: Full 50.000 ảnh với RandomCrop, RandomHorizontalFlip, Normalize.
    - TestEvalLoader: test_eval_size ảnh test cố định (không augmentation) để đánh giá nhanh mỗi epoch.
    - TestFullLoader: Toàn bộ 10.000 ảnh test.
    """
    mean = CIFAR10_MEAN
    std = CIFAR10_STD

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    trainset = CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    testset_eval = CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)
    testset_full = CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)

    # Tập con test cố định theo seed
    generator = torch.Generator().manual_seed(seed)
    test_perm = torch.randperm(len(testset_eval), generator=generator).tolist()
    test_eval_subset = Subset(testset_eval, test_perm[:test_eval_size])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_eval_loader = DataLoader(test_eval_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_full_loader = DataLoader(testset_full, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, test_eval_loader, test_full_loader


@torch.no_grad()
def evaluate_adversary(client, adv, loader, device, client_mode="eval"):
    """
    Đánh giá chất lượng tái tạo của Adversary trên loader:
    client_mode = 'eval': đo trên z với client.eval() (chế độ chuẩn tất định)
    client_mode = 'train': đo trên z với client.train() (đo độ lệch phân phối BatchNorm batch-level)
    """
    if client_mode == "eval":
        client.eval()
    else:
        client.train()
    adv.eval()

    total_samples = 0
    psnr_sum = 0.0
    ssim_sum = 0.0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        b = x.size(0)
        z = client(x)
        x_rec = adv(z)

        p, s = psnr_ssim(x, x_rec, CIFAR10_MEAN, CIFAR10_STD)
        psnr_sum += p * b
        ssim_sum += s * b
        total_samples += b

    return psnr_sum / max(total_samples, 1), ssim_sum / max(total_samples, 1)


@torch.no_grad()
def evaluate_task_acc(client, server, loader, device):
    """
    Đánh giá độ chính xác phân loại của Split Learning (Client + Server).
    """
    client.eval()
    server.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        z = client(x)
        logits = server(z)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return correct / max(total, 1)


def plot_b8_curves(history, save_path, lam, adv_type):
    """
    Vẽ biểu đồ hội tụ của B8:
    (1) Task Test Acc
    (2) Adversary SSIM (Eval vs Train vs Max)
    (3) Raw Rec MSE vs Cap & Clamp Ratio
    """
    epochs = [h["epoch"] for h in history]
    accs = [h["test_acc"] * 100 for h in history]
    ssim_eval = [h["adv_ssim_eval"] for h in history]
    ssim_train = [h["adv_ssim_train"] for h in history]
    raw_mse = [h["raw_rec_mse"] for h in history]
    clamp_pct = [h["clamp_ratio"] * 100 for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Task Acc
    axes[0].plot(epochs, accs, color="#1f77b4", linewidth=2.0, marker="o", markersize=3)
    axes[0].axhline(92.7, color="#d62728", linestyle="--", label="Ngưỡng 92.7% (B0 - 2%)")
    axes[0].set_title(f"Task Test Accuracy (B8-{adv_type}, $\\lambda={lam}$)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Adversary SSIM (Eval vs Train)
    axes[1].plot(epochs, ssim_eval, color="#2ca02c", linewidth=2.0, label="z (eval mode)", marker="^", markersize=3)
    axes[1].plot(epochs, ssim_train, color="#ff7f0e", linewidth=1.8, linestyle="--", label="z (train mode)", marker="v", markersize=3)
    axes[1].set_title("Adversary SSIM (Nội bộ lúc Train)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SSIM")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Raw Rec MSE & Clamp Ratio
    ax3 = axes[2]
    line1 = ax3.plot(epochs, raw_mse, color="#9467bd", linewidth=2.0, label="Raw Rec MSE")
    line2 = ax3.axhline(1.0, color="gray", linestyle=":", label="Cap = 1.0")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("MSE Loss", color="#9467bd")

    ax3_twin = ax3.twinx()
    line3 = ax3_twin.plot(epochs, clamp_pct, color="#e377c2", linewidth=1.5, linestyle="-.", label="Clamp Ratio (%)")
    ax3_twin.set_ylabel("Clamp Ratio (%)", color="#e377c2")
    ax3.set_title("Reconstruction MSE & Tỉ lệ Clamped", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)

    lines = line1 + [line2] + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def train_single_b8(lam, args, device):
    scenario_id = f"b8_{args.adversary}_lam{lam}_s{args.seed}"
    print("\n" + "=" * 90)
    print(f"BẮT ĐẦU HUẤN LUYỆN B8: {scenario_id.upper()}")
    print(f"Cấu hình: Adversary={args.adversary.upper()} | Lambda={lam} | Cap={args.cap} | Seed={args.seed}")
    print(f"Huấn luyện Split Learning 100 Epochs (MultiStepLR [50, 75]) | Single Forward Pass")
    print("=" * 90, flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Nạp dữ liệu
    train_loader, test_eval_loader, test_full_loader = get_b8_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_eval_size=args.test_eval_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # 2. Kiểm chứng E[x^2] trên dữ liệu huấn luyện
    x_sq_sum = 0.0
    n_sq_batches = 0
    with torch.no_grad():
        for x_b, _ in train_loader:
            x_b = x_b.to(device)
            x_sq_sum += (x_b ** 2).mean().item()
            n_sq_batches += 1
            if n_sq_batches >= 20:
                break
    avg_x_sq = x_sq_sum / max(n_sq_batches, 1)
    print(f"[KIỂM CHỨNG CAP] E[x^2] (pixel sau chuẩn hóa): {avg_x_sq:.4f} | Cap đang áp dụng: {args.cap:.4f}")

    # 3. Khởi tạo Client, Server, và Adversary
    client = ClientModel().to(device)
    server = ServerModel().to(device)

    if args.adversary == "conv":
        adv = Decoder(in_channels=64, out_channels=3).to(device)
    elif args.adversary == "mlp":
        adv = LearnedAdaptiveDecoderB8(in_features=65536, D=65536, hidden_dim=512).to(device)
    else:
        raise ValueError(f"Không hỗ trợ loại adversary: {args.adversary}")

    # Optimizers
    opt_cs = torch.optim.SGD(
        list(client.parameters()) + list(server.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    sched_cs = MultiStepLR(opt_cs, milestones=[50, 75], gamma=0.1)

    opt_adv = torch.optim.Adam(adv.parameters(), lr=args.adv_lr)

    criterion_task = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    history = []
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        client.train()
        server.train()
        adv.train()

        total_task_loss = 0.0
        total_raw_rec = 0.0
        n_clamped_batches = 0
        total_batches = 0
        train_correct = 0
        train_samples = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            b = x.size(0)

            # [QUAN TRỌNG] Client forward ĐÚNG 1 LẦN DUY NHẤT để không nhân đôi thống kê BatchNorm
            z = client(x)

            # -------------------------------------------------------------
            # BƯỚC 1: MAX (Adversary học giải mã z hiện tại)
            # -------------------------------------------------------------
            z_det = z.detach()
            x_rec_adv = adv(z_det)
            loss_adv = criterion_mse(x_rec_adv, x)

            opt_adv.zero_grad()
            loss_adv.backward()
            opt_adv.step()

            # -------------------------------------------------------------
            # BƯỚC 2: MIN (Client và Server tối ưu đối kháng)
            # -------------------------------------------------------------
            # [QUAN TRỌNG] Tạm thời tắt requires_grad của adv để:
            # (1) Chống rò rỉ gradient vào adv
            # (2) Tăng tốc tính toán backward (không tính gradient tham số adv)
            for p in adv.parameters():
                p.requires_grad = False

            logits = server(z)
            loss_task = criterion_task(logits, y)

            # Tái tạo qua adv (đồ thị đạo hàm chảy ngược từ adv về z và client)
            x_rec_client = adv(z)
            rec_mse = criterion_mse(x_rec_client, x)

            # Kiểm tra xem batch này có bị clamp chặn hay không
            if rec_mse.item() >= args.cap:
                n_clamped_batches += 1

            loss_total = loss_task - lam * torch.clamp(rec_mse, max=args.cap)

            opt_cs.zero_grad()
            loss_total.backward()
            opt_cs.step()

            # Bật lại requires_grad cho adv cho batch kế tiếp
            for p in adv.parameters():
                p.requires_grad = True

            total_task_loss += loss_task.item() * b
            total_raw_rec += rec_mse.item()
            total_batches += 1
            train_correct += (logits.argmax(1) == y).sum().item()
            train_samples += b

        sched_cs.step()

        avg_task_loss = total_task_loss / train_samples
        train_acc = train_correct / train_samples
        avg_raw_rec = total_raw_rec / total_batches
        clamp_ratio = n_clamped_batches / total_batches

        # -------------------------------------------------------------
        # ĐÁNH GIÁ ĐỊNH KỲ MỖI EPOCH (TRÊN 2.000 ẢNH TEST CỐ ĐỊNH)
        # -------------------------------------------------------------
        should_eval = (ep == 1 or ep % args.eval_freq == 0 or ep == args.epochs)
        if should_eval:
            # 1. Task Test Accuracy
            test_acc = evaluate_task_acc(client, server, test_eval_loader, device)

            # 2. Adversary Reconstruction: Đo trên cả z_eval và z_train
            psnr_eval, ssim_eval = evaluate_adversary(client, adv, test_eval_loader, device, client_mode="eval")
            psnr_train, ssim_train = evaluate_adversary(client, adv, test_eval_loader, device, client_mode="train")

            # Lấy giá trị cao hơn thận trọng
            max_ssim = max(ssim_eval, ssim_train)
            max_psnr = max(psnr_eval, psnr_train)

            # 3. Tính dCor định kỳ (mỗi 10 epochs hoặc epoch cuối)
            dcor_val = None
            if ep % 10 == 0 or ep == args.epochs:
                dcor_sum = 0.0
                n_dcor_b = 0
                client.eval()
                with torch.no_grad():
                    for x_dc, _ in test_eval_loader:
                        x_dc = x_dc.to(device)
                        z_dc = client(x_dc)
                        dcor_sum += distance_correlation(x_dc, z_dc).item()
                        n_dcor_b += 1
                        if n_dcor_b >= 10:
                            break
                dcor_val = round(dcor_sum / max(n_dcor_b, 1), 4)

            record = {
                "epoch": ep,
                "train_task_loss": round(avg_task_loss, 4),
                "train_acc": round(train_acc * 100, 2),
                "test_acc": round(test_acc * 100, 2),
                "raw_rec_mse": round(avg_raw_rec, 4),
                "clamp_ratio": round(clamp_ratio, 4),
                "adv_psnr_eval": round(psnr_eval, 2),
                "adv_ssim_eval": round(ssim_eval, 4),
                "adv_psnr_train": round(psnr_train, 2),
                "adv_ssim_train": round(ssim_train, 4),
                "adv_ssim_max": round(max_ssim, 4),
                "adv_psnr_max": round(max_psnr, 2),
                "dcor": dcor_val,
            }
            history.append(record)

            dcor_str = f" | dCor: {dcor_val}" if dcor_val is not None else ""
            print(
                f"  [Ep {ep:3d}/{args.epochs}] Task Acc: {test_acc*100:.2f}% | "
                f"Raw Rec: {avg_raw_rec:.4f} (Clamp: {clamp_ratio*100:.1f}%) | "
                f"Adv Eval: {psnr_eval:.2f}dB / {ssim_eval:.4f} | "
                f"Adv Train: {psnr_train:.2f}dB / {ssim_train:.4f} (Max SSIM: {max_ssim:.4f}){dcor_str}",
                flush=True
            )

    t_total = time.time() - t_start
    print(f"\n[HOÀN TẤT HUẤN LUYỆN] Xong sau {t_total:.1f}s!", flush=True)

    # Đánh giá cuối cùng trên Full Test Set (10.000 ảnh)
    final_test_acc = evaluate_task_acc(client, server, test_full_loader, device)
    final_psnr_eval, final_ssim_eval = evaluate_adversary(client, adv, test_full_loader, device, client_mode="eval")
    final_psnr_train, final_ssim_train = evaluate_adversary(client, adv, test_full_loader, device, client_mode="train")
    final_max_ssim = max(final_ssim_eval, final_ssim_train)
    final_max_psnr = max(final_psnr_eval, final_psnr_train)

    # Lưu checkpoint
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(client.state_dict(), os.path.join(ckpt_dir, f"{scenario_id}_client.pt"))
    torch.save(server.state_dict(), os.path.join(ckpt_dir, f"{scenario_id}_server.pt"))
    torch.save(adv.state_dict(), os.path.join(ckpt_dir, f"{scenario_id}_adv.pt"))

    # Vẽ đồ thị
    curve_path = os.path.join(args.output_dir, f"curve_{scenario_id}.png")
    plot_b8_curves(history, curve_path, lam, args.adversary)

    # Tính trung bình 10 epoch cuối của adversary
    last10_ssim_max = [h["adv_ssim_max"] for h in history[-10:]]
    last10_psnr_max = [h["adv_psnr_max"] for h in history[-10:]]
    mean_adv_ssim_10 = float(np.mean(last10_ssim_max))
    mean_adv_psnr_10 = float(np.mean(last10_psnr_max))

    res = {
        "scenario_id": scenario_id,
        "adversary_type": args.adversary,
        "lambda": lam,
        "cap": args.cap,
        "seed": args.seed,
        "task_test_acc": round(final_test_acc * 100, 2),
        "mean_adv_ssim_last10": round(mean_adv_ssim_10, 4),
        "mean_adv_psnr_last10": round(mean_adv_psnr_10, 2),
        "final_adv_ssim_eval": round(final_ssim_eval, 4),
        "final_adv_ssim_train": round(final_ssim_train, 4),
        "final_adv_ssim_max": round(final_max_ssim, 4),
        "final_adv_psnr_max": round(final_max_psnr, 2),
        "training_time_s": round(t_total, 1),
        "meets_utility_criterion": bool((final_test_acc * 100) >= 92.7),
        "history": history,
    }

    return res


def main():
    parser = argparse.ArgumentParser(description="Baseline B8: Adversarial Training (Min-Max Privacy Defense)")
    parser.add_argument("--adversary", type=str, default="conv", choices=["conv", "mlp"],
                        help="Kiến trúc adversary trong lúc train: 'conv' (ConvDecoder) hoặc 'mlp' (LearnedAdaptiveDecoderB8)")
    parser.add_argument("--lambdas", type=str, default="0.1,0.5,1.0,5.0",
                        help="Danh sách hệ số phạt đối kháng lambda cần quét (mặc định: '0.1,0.5,1.0,5.0')")
    parser.add_argument("--cap", type=float, default=1.0,
                        help="Cận trên chặn MSE reconstruction loss (mặc định: 1.0, tương ứng E[x^2])")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (mặc định: 42)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện Split Learning (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate cho SGD (mặc định: 0.1)")
    parser.add_argument("--adv-lr", type=float, default=1e-3, help="Learning rate Adam cho Adversary (mặc định: 1e-3)")
    parser.add_argument("--eval-freq", type=int, default=1, help="Tần suất đánh giá mỗi N epoch (mặc định: 1)")
    parser.add_argument("--test-eval-size", type=int, default=2000, help="Kích thước tập test nhanh để log (mặc định: 2000)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step3_B8"))
    parser.add_argument("--backup-dir", type=str, default=None, help="Thư mục sao lưu Google Drive nếu chạy Colab")
    parser.add_argument("--resume", action="store_true", default=False, help="Bỏ qua các lambda đã hoàn thành")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Chạy thử 2 epochs để kiểm thử mã")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 2
        args.eval_freq = 1
        args.lambdas = "0.1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)

    lambda_list = [float(l.strip()) for l in args.lambdas.split(",") if l.strip()]

    print("=" * 90)
    print("BASELINE B8: ADVERSARIAL TRAINING (MIN-MAX GAME)")
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Kiến trúc Adversary: B8-{args.adversary.upper()}")
    print(f"Danh sách Lambda quét: {lambda_list}")
    print(f"Ngưỡng chọn Utility: Task Test Acc >= 92.7% (mất <= 2% so với B0 94.71%)")
    print(f"Thư mục kết quả: {args.output_dir}")
    print("=" * 90, flush=True)

    json_path = os.path.join(args.output_dir, f"results_b8_{args.adversary}_sweep.json")
    csv_path = os.path.join(args.output_dir, f"results_b8_{args.adversary}_sweep.csv")

    all_results = {}
    if args.resume and os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"[RESUME] Đã nạp {len(all_results)} kết quả lambda trước đó từ JSON.")
        except Exception:
            all_results = {}

    for lam in lambda_list:
        scenario_id = f"b8_{args.adversary}_lam{lam}_s{args.seed}"
        if args.resume and scenario_id in all_results:
            print(f"[BỎ QUA] {scenario_id} đã có kết quả.")
            continue

        res = train_single_b8(lam, args, device)
        all_results[scenario_id] = res

        # Cập nhật JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Cập nhật CSV
        with open(csv_path, "w", encoding="utf-8") as f:
            header = [
                "scenario_id", "adversary", "lambda", "cap", "seed",
                "task_test_acc", "mean_adv_ssim_last10", "mean_adv_psnr_last10",
                "final_adv_ssim_eval", "final_adv_ssim_train", "final_adv_ssim_max",
                "final_adv_psnr_max", "training_time_s", "meets_utility_criterion"
            ]
            f.write(",".join(header) + "\n")
            for r in all_results.values():
                row = [
                    r["scenario_id"],
                    r["adversary_type"],
                    str(r["lambda"]),
                    str(r["cap"]),
                    str(r["seed"]),
                    str(r["task_test_acc"]),
                    str(r["mean_adv_ssim_last10"]),
                    str(r["mean_adv_psnr_last10"]),
                    str(r["final_adv_ssim_eval"]),
                    str(r["final_adv_ssim_train"]),
                    str(r["final_adv_ssim_max"]),
                    str(r["final_adv_psnr_max"]),
                    str(r["training_time_s"]),
                    str(r["meets_utility_criterion"])
                ]
                f.write(",".join(row) + "\n")

        # Đồng bộ Google Drive nếu có
        if args.backup_dir and os.path.exists(args.backup_dir):
            import shutil
            shutil.copy2(json_path, os.path.join(args.backup_dir, os.path.basename(json_path)))
            shutil.copy2(csv_path, os.path.join(args.backup_dir, os.path.basename(csv_path)))
            curve_file = os.path.join(args.output_dir, f"curve_{scenario_id}.png")
            if os.path.exists(curve_file):
                shutil.copy2(curve_file, os.path.join(args.backup_dir, os.path.basename(curve_file)))
            print(f"[DRIVE SYNC] Đã đồng bộ kết quả Lambda {lam} sang Google Drive: {args.backup_dir}")

    # ==============================================================================
    # BẢNG TỔNG HỢP VÀ CHỌN LAMBDA TỐI ƯU
    # ==============================================================================
    print("\n" + "=" * 95)
    print(f"TỔNG HỢP QUÉT LAMBDA CHO B8-{args.adversary.upper()}:")
    print("=" * 95)
    print(f"{'Lambda':<8} | {'Task Acc (%)':<14} | {'Adv SSIM (Last10)':<20} | {'Adv SSIM (Eval/Train)':<24} | {'Đạt ngưỡng Acc >= 92.7%?'}")
    print("-" * 95)

    valid_candidates = []
    for r in all_results.values():
        lam = r["lambda"]
        acc = r["task_test_acc"]
        s_last10 = r["mean_adv_ssim_last10"]
        s_eval = r["final_adv_ssim_eval"]
        s_train = r["final_adv_ssim_train"]
        meets = r["meets_utility_criterion"]

        status_str = " THỎA MÃN" if meets else "❌ KHÔNG ĐẠT"
        print(f"{lam:<8} | {acc:<14.2f} | {s_last10:<20.4f} | {s_eval:.4f} / {s_train:.4f}{'':<11} | {status_str}")

        if meets:
            valid_candidates.append(r)

    print("-" * 95)
    if valid_candidates:
        # Chọn lambda có SSIM thấp nhất trong số các ứng viên thỏa mãn utility
        best_candidate = min(valid_candidates, key=lambda x: x["mean_adv_ssim_last10"])
        print(f"\n[ĐIỂM VẬN HÀNH TỐI ƯU ĐƯỢC CHỌN]:")
        print(f"  -> Lambda* = {best_candidate['lambda']}")
        print(f"  -> Task Test Acc = {best_candidate['task_test_acc']:.2f}% (>= 92.7%)")
        print(f"  -> Adversary SSIM (Nội bộ) = {best_candidate['mean_adv_ssim_last10']:.4f}")
        print(f"  -> Checkpoint Client: checkpoints/b8_{args.adversary}_lam{best_candidate['lambda']}_s{args.seed}_client.pt")
        print(f"  ==> Bước tiếp theo: Dùng Client đóng băng này để chạy 4 Attacker hậu kiểm!")
    else:
        print("\n[CẢNH BÁO]: Không có lambda nào thỏa mãn điều kiện Task Acc >= 92.7%. Báo cáo toàn bộ đường cong trade-off.")
    print("=" * 95 + "\n", flush=True)


if __name__ == "__main__":
    main()
