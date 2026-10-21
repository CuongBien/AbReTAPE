#!/usr/bin/env python3
# ==============================================================================
# AR-TAPE v0: RES-SFL-LITE (LEARNED BOTTLENECK + MIN-MAX STRONG ADVERSARY)
# ==============================================================================
# - Testbed: CIFAR-10, Split Learning
# - Client: ResNet-18 stem + layer1 + learned bottleneck (D=65536 -> k=1024)
# - Server: ProjectionMLP (k=1024 -> m=512 -> D=65536) + ResNet-18 layer2..4 + fc
# - In-training Adversary: LearnedAdaptiveDecoder (~34M params: MLP 1024->512->65536 + ConvDecoder)
# - Alternating Min-Max Loop:
#     * Single Forward Pass: Client forward đúng 1 lần mỗi batch để chống lệch BatchNorm
#     * Bước Max: Adversary học giải mã zt_det (Adam lr 1e-3)
#     * Bước Min: Client + Server tối ưu L_CE - lam * clamp(rec_mse, max=cap)
#     * Gradient isolation: Tắt requires_grad của Adversary ở bước Min
# - Cấu hình chuẩn: cap=1.5586, Adam lr=1e-3, SGD lr=0.1, MultiStepLR [50, 75], 100 epochs, seed 42
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
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
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
from src.data.cifar import CIFAR10_MEAN, CIFAR10_STD
from src.attacks import Decoder
from src.metrics import psnr_ssim, distance_correlation
from src.utils.visualize import save_reconstruction_grid


# ==============================================================================
# 1. KIẾN TRÚC MÔ HÌNH AR-TAPE v0
# ==============================================================================

class ClientWithBottleneck(nn.Module):
    """
    Client AR-TAPE v0 / v1:
    - Base: stem (Conv3->64, BN, ReLU) + layer1 (2 BasicBlocks, 64 channels) -> shape (B, 64, 32, 32)
    - Flatten: (B, 65536)
    - Bottleneck:
        * fixed_bottleneck=True: Phép chiếu trực giao cố định R in R^{D x k} (B7 mode F style, QR orthonormal)
        * fixed_bottleneck=False: Linear(D, k, bias=False) học được, khởi tạo trực chuẩn bằng QR.
    """
    def __init__(self, k=1024, D=65536, init_ortho=True, seed=42, fixed_bottleneck=False):
        super().__init__()
        self.base_client = ClientModel()
        self.k = k
        self.D = D
        self.fixed_bottleneck = fixed_bottleneck

        g = torch.Generator().manual_seed(seed)
        A = torch.randn(D, k, generator=g)
        Q, _ = torch.linalg.qr(A, mode="reduced")  # Q in R^{D x k}

        if fixed_bottleneck:
            self.register_buffer("R", Q)
            self.bottleneck = None
        else:
            self.R = None
            self.bottleneck = nn.Linear(D, k, bias=False)
            if init_ortho:
                with torch.no_grad():
                    self.bottleneck.weight.copy_(Q.t())  # weight shape: [k, D]

    def get_bottleneck_weight(self):
        """Trả về ma trận trọng số chiếu dạng [k, D] để dùng cho Conv lift attacker."""
        if self.fixed_bottleneck:
            return self.R.t()
        return self.bottleneck.weight

    def forward(self, x):
        h = self.base_client(x)
        h_flat = h.view(h.size(0), -1)
        if self.fixed_bottleneck:
            zt = torch.matmul(h_flat, self.R)
        else:
            zt = self.bottleneck(h_flat)
        return zt

    def forward_features(self, x):
        h = self.base_client(x)
        h_flat = h.view(h.size(0), -1)
        if self.fixed_bottleneck:
            zt = torch.matmul(h_flat, self.R)
        else:
            zt = self.bottleneck(h_flat)
        return zt, h

    def load_state_dict(self, state_dict, strict=True):
        clean_sd = {}
        for k, v in state_dict.items():
            clean_sd[k] = v
        # Tương thích chéo: nếu state_dict có 'R' nhưng model dùng 'bottleneck.weight'
        if "R" in clean_sd and hasattr(self, "bottleneck") and self.bottleneck is not None:
            self.bottleneck.weight.data.copy_(clean_sd["R"].t())
            clean_sd.pop("R", None)
        # Hoặc nếu state_dict có 'bottleneck.weight' nhưng model dùng 'R'
        elif "bottleneck.weight" in clean_sd and hasattr(self, "R") and self.R is not None:
            self.R.data.copy_(clean_sd["bottleneck.weight"].t())
            clean_sd.pop("bottleneck.weight", None)
        return super().load_state_dict(clean_sd, strict=False)


class ServerWithBottleneck(nn.Module):
    """
    Server AR-TAPE v0 / v1:
    - unprojector_mode:
        * 'linear_fixed': Toán tử nâng chiều trực giao R^T (0 tham số thừa, B7 mode F style).
                          z_hat = z_t @ R^T in R^{B x D} -> không làm hẹp rank, không trôi BatchNorm.
        * 'linear_learned': Tầng Linear(k, D, bias=False) học được, khởi tạo trực giao từ R^T.
        * 'mlp': Mạng MLP phi tuyến Linear(k, m) -> BN -> ReLU -> Linear(m, D) (B7 mode L style).
    - Reshape: (B, 64, 32, 32)
    - Base Server: ResNet-18 layer2..4 + AvgPool + FC (10 classes)
    """
    def __init__(self, k=1024, m=512, D=65536, unprojector_mode="linear_fixed", ortho_matrix=None, seed=42):
        super().__init__()
        self.k = k
        self.m = m
        self.D = D
        self.unprojector_mode = unprojector_mode

        if unprojector_mode == "linear_fixed":
            if ortho_matrix is not None:
                self.register_buffer("R", ortho_matrix.clone())
            else:
                g = torch.Generator().manual_seed(seed)
                A = torch.randn(D, k, generator=g)
                Q, _ = torch.linalg.qr(A, mode="reduced")
                self.register_buffer("R", Q)
            self.unprojector = None
        elif unprojector_mode == "linear_learned":
            self.unprojector = nn.Linear(k, D, bias=False)
            if ortho_matrix is not None:
                Q = ortho_matrix
            else:
                g = torch.Generator().manual_seed(seed)
                A = torch.randn(D, k, generator=g)
                Q, _ = torch.linalg.qr(A, mode="reduced")
            with torch.no_grad():
                self.unprojector.weight.copy_(Q)  # [D, k] in PyTorch Linear(k, D)
        elif unprojector_mode == "mlp":
            self.unprojector = nn.Sequential(
                nn.Linear(k, m),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True),
                nn.Linear(m, D)
            )
        else:
            raise ValueError(f"Không hỗ trợ unprojector_mode='{unprojector_mode}'. Chọn 'linear_fixed', 'linear_learned', hoặc 'mlp'.")

        self.base_server = ServerModel()

    def lift(self, zt):
        if zt.dim() > 2:
            zt = zt.view(zt.size(0), -1)
        if self.unprojector_mode == "linear_fixed":
            zh = torch.matmul(zt, self.R.t())
        else:
            zh = self.unprojector(zt)
        return zh

    def forward(self, zt):
        zh = self.lift(zt)
        zh_tensor = zh.view(zt.size(0), 64, 32, 32)
        return self.base_server(zh_tensor)

    def load_state_dict(self, state_dict, strict=True):
        clean_sd = {}
        for k, v in state_dict.items():
            clean_sd[k] = v
        if "R" in clean_sd and hasattr(self, "unprojector") and self.unprojector is not None and isinstance(self.unprojector, nn.Linear):
            self.unprojector.weight.data.copy_(clean_sd["R"])
            clean_sd.pop("R", None)
        elif "unprojector.weight" in clean_sd and hasattr(self, "R") and self.R is not None:
            self.R.data.copy_(clean_sd["unprojector.weight"])
            clean_sd.pop("unprojector.weight", None)
        return super().load_state_dict(clean_sd, strict=False)


class LearnedAdaptiveDecoder(nn.Module):
    """
    In-training Strong Adversary (~34M parameters):
    zt in R^{B x k} -> unprojector (Linear(k, 512) -> BN -> ReLU -> Linear(512, 65536) -> BN)
    -> reshape (B, 64, 32, 32) -> ConvDecoder(64, 3) -> x_rec in R^{B x 3 x 32 x 32}
    """
    def __init__(self, k=1024, D=65536, hidden_dim=512):
        super().__init__()
        self.k = k
        self.D = D
        self.unprojector = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, D),
            nn.BatchNorm1d(D)
        )
        self.conv_decoder = Decoder(in_channels=64, out_channels=3)

    def forward(self, zt):
        if zt.dim() > 2:
            zt = zt.view(zt.size(0), -1)
        z_unproj = self.unprojector(zt).view(-1, 64, 32, 32)
        return self.conv_decoder(z_unproj)


# ==============================================================================
# 2. TIỆN ÍCH DỮ LIỆU & ĐÁNH GIÁ
# ==============================================================================

def get_data_loaders(data_dir="./data", batch_size=128, test_eval_size=2000, num_workers=0, seed=42):
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
    if client_mode == "eval":
        client.eval()
    else:
        client.train()
    adv.eval()

    total_samples = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    criterion = nn.MSELoss()

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        b = x.size(0)
        zt = client(x)
        x_rec = adv(zt)

        mse_val = criterion(x_rec, x).item()
        p, s = psnr_ssim(x, x_rec, CIFAR10_MEAN, CIFAR10_STD)
        mse_sum += mse_val * b
        psnr_sum += p * b
        ssim_sum += s * b
        total_samples += b

    n = max(total_samples, 1)
    return psnr_sum / n, ssim_sum / n, mse_sum / n


@torch.no_grad()
def evaluate_task_acc(client, server, loader, device, mode="eval"):
    if mode == "eval":
        client.eval()
        server.eval()
    else:
        client.train()
        server.train()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        zt = client(x)
        logits = server(zt)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return correct / max(total, 1)


def plot_artape_v0_curves(history, save_path, lam):
    epochs = [h["epoch"] for h in history]
    accs_eval = [h["test_acc_eval"] for h in history]
    accs_train = [h.get("test_acc_train", h["test_acc_eval"]) for h in history]
    ssim_eval = [h["adv_ssim_eval"] for h in history]
    ssim_train = [h["adv_ssim_train"] for h in history]
    raw_mse = [h["raw_rec_mse"] for h in history]
    clamp_pct = [h["clamp_ratio"] * 100 for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Task Accuracy
    axes[0].plot(epochs, accs_eval, color="#1f77b4", linewidth=2.0, marker="o", markersize=3, label="Test Acc (eval mode)")
    axes[0].plot(epochs, accs_train, color="#17becf", linewidth=1.5, linestyle="--", label="Test Acc (train mode)")
    axes[0].axhline(92.7, color="#d62728", linestyle="--", label="Ngưỡng PoC 92.7% (B0 - 2%)")
    axes[0].set_title(f"Task Test Accuracy (AR-TAPE v0, $\\lambda={lam}$)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Adversary SSIM
    axes[1].plot(epochs, ssim_eval, color="#2ca02c", linewidth=2.0, label="z (eval mode)", marker="^", markersize=3)
    axes[1].plot(epochs, ssim_train, color="#ff7f0e", linewidth=1.8, linestyle="--", label="z (train mode)", marker="v", markersize=3)
    axes[1].axhline(0.30, color="#d62728", linestyle=":", label="Mục tiêu Privacy SSIM < 0.3")
    axes[1].set_title("In-training Adversary SSIM", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SSIM")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Rec MSE & Clamping
    ax3 = axes[2]
    line1 = ax3.plot(epochs, raw_mse, color="#9467bd", linewidth=2.0, label="Raw Rec MSE")
    line2 = ax3.axhline(1.5586, color="gray", linestyle=":", label="Cap = 1.5586")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("MSE Loss", color="#9467bd")

    ax3_twin = ax3.twinx()
    line3 = ax3_twin.plot(epochs, clamp_pct, color="#e377c2", linewidth=1.5, linestyle="-.", label="Clamp Ratio (%)")
    ax3_twin.set_ylabel("Clamp Ratio (%)", color="#e377c2")
    ax3.set_title("Reconstruction MSE & Clamp Ratio", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)

    lines = line1 + [line2] + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 3. VÒNG LẶP HUẤN LUYỆN MIN-MAX CHÍNH
# ==============================================================================

def train_single_artape_v0(lam, args, device):
    mode_str = f"_{args.unprojector_mode}" if args.unprojector_mode != "mlp" else ""
    fix_str = "_fixR" if getattr(args, "fixed_bottleneck", False) else ""
    scenario_id = f"artape_v0_k{args.k}{mode_str}{fix_str}_lam{lam}_s{args.seed}"
    print("\n" + "=" * 90)
    print(f"BẮT ĐẦU HUẤN LUYỆN AR-TAPE: {scenario_id.upper()}")
    print(f"Cấu hình: k={args.k} | Unprojector Mode={args.unprojector_mode} | Fixed Bottleneck={getattr(args, 'fixed_bottleneck', False)}")
    print(f"Lambda={lam} (Warmup={getattr(args, 'warmup_epochs', 0)}) | Cap={args.cap} | Seed={args.seed}")
    print(f"Client: Stem+Layer1 + Bottleneck({args.D}->{args.k}) | Server: Mode {args.unprojector_mode} + ServerModel")
    print(f"Adversary: LearnedAdaptiveDecoder ({args.k}->{args.m}->{args.D} + ConvDecoder, ~34M params)")
    print(f"Huấn luyện Split Learning 100 Epochs (MultiStepLR [50, 75]) | Single Forward Pass")
    print("=" * 90, flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, test_eval_loader, test_full_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_eval_size=args.test_eval_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # 1. Khởi tạo Client, Server, Adversary
    client = ClientWithBottleneck(
        k=args.k, D=args.D, init_ortho=True, seed=args.seed, fixed_bottleneck=getattr(args, "fixed_bottleneck", False)
    ).to(device)
    ortho_q = client.R if client.fixed_bottleneck else client.bottleneck.weight.t().detach()
    server = ServerWithBottleneck(
        k=args.k, m=args.m, D=args.D, unprojector_mode=args.unprojector_mode, ortho_matrix=ortho_q, seed=args.seed
    ).to(device)
    adv = LearnedAdaptiveDecoder(k=args.k, D=args.D, hidden_dim=args.m).to(device)

    # 2. Optimizers & Schedulers
    trainable_cs = [p for p in client.parameters() if p.requires_grad] + [p for p in server.parameters() if p.requires_grad]
    opt_cs = torch.optim.SGD(
        trainable_cs,
        lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    if getattr(args, "scheduler", "multistep") == "cosine":
        sched_cs = CosineAnnealingLR(opt_cs, T_max=args.epochs)
    else:
        sched_cs = MultiStepLR(opt_cs, milestones=[50, 75], gamma=0.1)
    opt_adv = torch.optim.Adam(adv.parameters(), lr=args.adv_lr)

    criterion_task = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    history = []
    t_start = time.time()
    best_task_acc = 0.0

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

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
        total_norm_priv = 0.0
        total_norm_task = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            b = x.size(0)

            # [QUAN TRỌNG] Single forward pass: Client forward đúng 1 lần
            zt = client(x)

            # -------------------------------------------------------------
            # HUẤN LUYỆN SPLIT LEARNING
            # -------------------------------------------------------------
            if lam > 0.0:
                # BƯỚC 1: MAX (Adversary học giải mã zt hiện tại)
                zt_det = zt.detach()
                x_rec_adv = adv(zt_det)
                loss_adv = criterion_mse(x_rec_adv, x)

                opt_adv.zero_grad()
                loss_adv.backward()
                opt_adv.step()

                # BƯỚC 2: MIN (Client và Server tối ưu đối kháng)
                for p in adv.parameters():
                    p.requires_grad = False

                logits = server(zt)
                loss_task = criterion_task(logits, y)

                # Tái tạo qua adv (đồ thị đạo hàm chảy ngược từ adv về zt và client)
                x_rec_client = adv(zt)
                rec_mse = criterion_mse(x_rec_client, x)

                if rec_mse.item() >= args.cap:
                    n_clamped_batches += 1

                if hasattr(args, "warmup_epochs") and args.warmup_epochs > 0:
                    cur_lam = lam * min(1.0, ep / max(args.warmup_epochs, 1))
                else:
                    cur_lam = lam

                # Lan truyền ngược Task Loss
                opt_cs.zero_grad()
                loss_task.backward(retain_graph=True)

                client_params = [p for p in client.parameters() if p.requires_grad]
                norm_task = torch.sqrt(sum((p.grad.norm(2) ** 2) for p in client_params if p.grad is not None)).item()

                # Tính riêng gradient đối kháng trên Client
                loss_priv = - cur_lam * torch.clamp(rec_mse, max=args.cap)
                grads_priv = torch.autograd.grad(loss_priv, client_params, retain_graph=False)

                norm_priv = torch.sqrt(sum((g.norm(2) ** 2) for g in grads_priv if g is not None)).item()

                # Cắt chuẩn gradient đối kháng nếu có cấu hình
                if hasattr(args, "clip_rec_grad") and args.clip_rec_grad is not None and args.clip_rec_grad > 0:
                    clip_coef = min(1.0, args.clip_rec_grad / (norm_priv + 1e-6))
                else:
                    clip_coef = 1.0

                for p, g in zip(client_params, grads_priv):
                    if p.grad is not None:
                        p.grad.add_(g, alpha=clip_coef)
                    else:
                        p.grad = g * clip_coef

                all_params = list(client.parameters()) + list(server.parameters())
                if hasattr(args, "clip_grad") and args.clip_grad is not None and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.clip_grad)

                opt_cs.step()

                for p in adv.parameters():
                    p.requires_grad = True

                raw_rec_val = rec_mse.item()
            else:
                # lam == 0: Huấn luyện thuần Split Learning qua bottleneck, không adversary
                logits = server(zt)
                loss_task = criterion_task(logits, y)

                opt_cs.zero_grad()
                loss_task.backward()

                all_params = list(client.parameters()) + list(server.parameters())
                if hasattr(args, "clip_grad") and args.clip_grad is not None and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.clip_grad)

                opt_cs.step()

                norm_task = 0.0
                norm_priv = 0.0
                raw_rec_val = 0.0

            total_task_loss += loss_task.item() * b
            total_raw_rec += raw_rec_val
            total_batches += 1
            train_correct += (logits.argmax(1) == y).sum().item()
            train_samples += b
            total_norm_priv += norm_priv
            total_norm_task += norm_task

        sched_cs.step()

        avg_task_loss = total_task_loss / train_samples
        train_acc = train_correct / train_samples
        avg_raw_rec = total_raw_rec / total_batches
        clamp_ratio = n_clamped_batches / total_batches
        avg_norm_priv = total_norm_priv / max(total_batches, 1)
        avg_norm_task = total_norm_task / max(total_batches, 1)

        # -------------------------------------------------------------
        # ĐÁNH GIÁ ĐỊNH KỲ MỖI EPOCH
        # -------------------------------------------------------------
        should_eval = (ep == 1 or ep % args.eval_freq == 0 or ep == args.epochs)
        if should_eval:
            test_acc_eval = evaluate_task_acc(client, server, test_eval_loader, device, mode="eval")
            test_acc_train = evaluate_task_acc(client, server, test_eval_loader, device, mode="train")

            psnr_eval, ssim_eval, mse_eval = evaluate_adversary(client, adv, test_eval_loader, device, client_mode="eval")
            psnr_train, ssim_train, mse_train = evaluate_adversary(client, adv, test_eval_loader, device, client_mode="train")

            max_ssim = max(ssim_eval, ssim_train)
            max_psnr = max(psnr_eval, psnr_train)

            # Đo dCor định kỳ (mỗi 10 epoch hoặc epoch cuối)
            dcor_val = None
            if ep % 10 == 0 or ep == args.epochs:
                dcor_sum = 0.0
                n_dcor_b = 0
                client.eval()
                with torch.no_grad():
                    for x_dc, _ in test_eval_loader:
                        x_dc = x_dc.to(device)
                        zt_dc = client(x_dc)
                        dcor_sum += distance_correlation(x_dc, zt_dc).item()
                        n_dcor_b += 1
                        if n_dcor_b >= 10:
                            break
                dcor_val = round(dcor_sum / max(n_dcor_b, 1), 4)

            # Đo trên toàn bộ 10k test ở các mốc quan trọng (epoch 50, 75, 90..100)
            full_test_acc = None
            if ep >= 90 or ep % 25 == 0 or ep == args.epochs:
                full_test_acc = evaluate_task_acc(client, server, test_full_loader, device, mode="eval")

            record = {
                "epoch": ep,
                "train_task_loss": round(avg_task_loss, 4),
                "train_acc": round(train_acc * 100, 2),
                "test_acc_eval": round(test_acc_eval * 100, 2),
                "test_acc_train": round(test_acc_train * 100, 2),
                "test_acc_full": round(full_test_acc * 100, 2) if full_test_acc is not None else None,
                "raw_rec_mse": round(avg_raw_rec, 4),
                "clamp_ratio": round(clamp_ratio, 4),
                "norm_task": round(avg_norm_task, 4),
                "norm_priv": round(avg_norm_priv, 4),
                "adv_psnr_eval": round(psnr_eval, 2),
                "adv_ssim_eval": round(ssim_eval, 4),
                "adv_mse_eval": round(mse_eval, 4),
                "adv_psnr_train": round(psnr_train, 2),
                "adv_ssim_train": round(ssim_train, 4),
                "adv_mse_train": round(mse_train, 4),
                "adv_ssim_max": round(max_ssim, 4),
                "adv_psnr_max": round(max_psnr, 2),
                "dcor": dcor_val
            }
            history.append(record)

            dcor_str = f" | dCor: {dcor_val:.4f}" if dcor_val is not None else ""
            full_acc_str = f" | FullTest: {full_test_acc*100:.2f}%" if full_test_acc is not None else ""
            print(f"  [Ep {ep:3d}/{args.epochs}] Acc(Eval/Train): {test_acc_eval*100:.2f}% / {test_acc_train*100:.2f}%{full_acc_str} | "
                  f"Norm(Task/Rec): {avg_norm_task:.3f} / {avg_norm_priv:.3f} | "
                  f"Adv Eval: {psnr_eval:.2f}dB / {ssim_eval:.4f} | Adv Train: {psnr_train:.2f}dB / {ssim_train:.4f} (Max: {max_ssim:.4f}){dcor_str}",
                  flush=True)

            if test_acc_eval > best_task_acc:
                best_task_acc = test_acc_eval
                torch.save({
                    "client": client.state_dict(),
                    "server": server.state_dict(),
                    "adv": adv.state_dict(),
                    "epoch": ep,
                    "test_acc": test_acc_eval,
                    "lambda": lam
                }, os.path.join(checkpoint_dir, f"{scenario_id}_best.pt"))

    train_time = time.time() - t_start

    # Đánh giá toàn diện epoch cuối trên toàn bộ 10.000 test
    final_full_acc = evaluate_task_acc(client, server, test_full_loader, device, mode="eval")
    final_full_acc_train = evaluate_task_acc(client, server, test_full_loader, device, mode="train")

    last10 = history[-10:] if len(history) >= 10 else history
    mean_task_acc_last10 = np.mean([h["test_acc_eval"] for h in last10])
    full_accs = [h["test_acc_full"] for h in last10 if h.get("test_acc_full") is not None]
    mean_full_acc_last10 = np.mean(full_accs) if full_accs else final_full_acc * 100
    mean_adv_ssim_last10 = np.mean([h["adv_ssim_max"] for h in last10])
    mean_adv_psnr_last10 = np.mean([h["adv_psnr_max"] for h in last10])

    # Lưu checkpoints cuối
    torch.save(client.state_dict(), os.path.join(checkpoint_dir, f"{scenario_id}_client.pt"))
    torch.save(server.state_dict(), os.path.join(checkpoint_dir, f"{scenario_id}_server.pt"))
    torch.save(adv.state_dict(), os.path.join(checkpoint_dir, f"{scenario_id}_adv.pt"))

    # Vẽ và lưu biểu đồ hội tụ
    plot_path = os.path.join(args.output_dir, f"curve_{scenario_id}.png")
    plot_artape_v0_curves(history, plot_path, lam)

    # Lưu ảnh tái tạo của adversary nội bộ
    grid_path = os.path.join(args.output_dir, f"rec_adv_{scenario_id}.png")
    save_reconstruction_grid(client, adv, test_eval_loader, device, CIFAR10_MEAN, CIFAR10_STD, save_path=grid_path, num_images=8)

    meets_utility = (final_full_acc * 100 >= 92.7)

    result = {
        "scenario_id": scenario_id,
        "k": args.k,
        "m": args.m,
        "lambda": lam,
        "cap": args.cap,
        "seed": args.seed,
        "final_full_test_acc": round(final_full_acc * 100, 2),
        "final_full_test_acc_train_mode": round(final_full_acc_train * 100, 2),
        "mean_full_test_acc_last10": round(float(mean_full_acc_last10), 2),
        "mean_task_acc_last10_2k": round(float(mean_task_acc_last10), 2),
        "mean_adv_ssim_last10": round(float(mean_adv_ssim_last10), 4),
        "mean_adv_psnr_last10": round(float(mean_adv_psnr_last10), 2),
        "final_adv_ssim_eval": history[-1]["adv_ssim_eval"],
        "final_adv_ssim_train": history[-1]["adv_ssim_train"],
        "final_adv_ssim_max": history[-1]["adv_ssim_max"],
        "final_adv_psnr_max": history[-1]["adv_psnr_max"],
        "final_dcor": history[-1]["dcor"],
        "unprojector_mode": args.unprojector_mode,
        "fixed_bottleneck": getattr(args, "fixed_bottleneck", False),
        "scheduler": getattr(args, "scheduler", "multistep"),
        "training_time_s": round(train_time, 1),
        "meets_utility_criterion": meets_utility,
        "history": history
    }

    print("\n" + "=" * 90)
    print(f"KẾT QUẢ AR-TAPE ({scenario_id.upper()}):")
    print(f"  - Full Test Acc (eval mode): {final_full_acc*100:.2f}% (Ngưỡng >= 92.7%: {'ĐẠT' if meets_utility else 'KHÔNG ĐẠT'})")
    print(f"  - Mean Last 10 Full Test Acc (10k): {mean_full_acc_last10:.2f}%")
    print(f"  - Mean Last 10 Fast Eval Acc (2k): {mean_task_acc_last10:.2f}%")
    print(f"  - In-train Adversary SSIM (eval/train): {history[-1]['adv_ssim_eval']:.4f} / {history[-1]['adv_ssim_train']:.4f} (Max: {history[-1]['adv_ssim_max']:.4f})")
    print(f"  - In-train Adversary PSNR: {history[-1]['adv_psnr_max']:.2f} dB")
    print(f"  - Final dCor: {history[-1]['dcor']}")
    print(f"  - Thời gian huấn luyện: {train_time:.1f}s")
    print("=" * 90 + "\n", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="AR-TAPE v0 / v1: ResSFL-lite (Learned/Ortho Bottleneck + Min-Max Strong Adversary)")
    parser.add_argument("--k", type=int, default=1024, help="Chiều không gian con bottleneck k (mặc định: 1024)")
    parser.add_argument("--m", type=int, default=512, help="Chiều ẩn MLP unprojector (mặc định: 512)")
    parser.add_argument("--D", type=int, default=65536, help="Số chiều phẳng smashed data (mặc định: 65536)")
    parser.add_argument("--unprojector-mode", type=str, default="linear_fixed", choices=["linear_fixed", "linear_learned", "mlp"], help="Chế độ unprojector tại server: linear_fixed (Mode F), linear_learned, hoặc mlp (Mode L)")
    parser.add_argument("--fixed-bottleneck", action="store_true", help="Cố định ma trận chiếu trực giao R tại client (không cập nhật gradient trên R, tương thích B7 Mode F)")
    parser.add_argument("--lambdas", type=str, default="0.1", help="Danh sách lambda phân tách bởi dấu phẩy (mặc định: '0.1')")
    parser.add_argument("--cap", type=float, default=1.5586, help="Cận trên chặn MSE loss (mặc định: 1.5586)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (mặc định: 42)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs Split Learning (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate SGD cho client & server (mặc định: 0.1)")
    parser.add_argument("--adv-lr", type=float, default=1e-3, help="Learning rate Adam cho adversary (mặc định: 1e-3)")
    parser.add_argument("--eval-freq", type=int, default=1, help="Tần suất đánh giá test (mặc định: 1)")
    parser.add_argument("--test-eval-size", type=int, default=2000, help="Kích thước tập test nhanh (mặc định: 2000)")
    parser.add_argument("--clip-grad", type=float, default=None, help="Cắt gradient norm tổng thể (mặc định: None)")
    parser.add_argument("--clip-rec-grad", type=float, default=None, help="Cắt riêng gradient norm của rec_mse trên client")
    parser.add_argument("--scheduler", type=str, default="multistep", choices=["multistep", "cosine"], help="Bộ điều chỉnh learning rate: 'multistep' ([50, 75]) hoặc 'cosine'")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Số epoch warmup lambda từ 0 (mặc định: 0)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_v0"))
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử 2 epochs để kiểm thử logic")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 2
        args.eval_freq = 1
        args.lambdas = "0.1"
        args.output_dir = os.path.join(args.output_dir, "dry_run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    lambda_list = [float(l.strip()) for l in args.lambdas.split(",") if l.strip()]

    print("=" * 90)
    print("KHỞI CHẠY THỰC NGHIỆM AR-TAPE (RES-SFL-LITE)")
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Bottleneck: {args.D} -> k={args.k} (Fixed={args.fixed_bottleneck}) | Server Unprojector: Mode {args.unprojector_mode}")
    print(f"Danh sách Lambda: {lambda_list} (Warmup={args.warmup_epochs}) | Seed: {args.seed} | Cap: {args.cap}")
    print(f"Thư mục lưu kết quả: {args.output_dir}")
    print("=" * 90, flush=True)

    json_path = os.path.join(args.output_dir, "results_artape_v0.json")
    csv_path = os.path.join(args.output_dir, "results_artape_v0.csv")

    all_results = {}
    for lam in lambda_list:
        res = train_single_artape_v0(lam, args, device)
        all_results[res["scenario_id"]] = res

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        with open(csv_path, "w", encoding="utf-8") as f:
            header = [
                "scenario_id", "k", "m", "lambda", "cap", "seed",
                "final_full_test_acc", "final_full_test_acc_train_mode",
                "mean_full_test_acc_last10", "mean_task_acc_last10_2k",
                "mean_adv_ssim_last10", "mean_adv_psnr_last10",
                "final_adv_ssim_eval", "final_adv_ssim_train", "final_adv_ssim_max",
                "final_adv_psnr_max", "final_dcor", "training_time_s", "meets_utility_criterion"
            ]
            f.write(",".join(header) + "\n")
            for r in all_results.values():
                row = [
                    r["scenario_id"], str(r["k"]), str(r["m"]), str(r["lambda"]), str(r["cap"]), str(r["seed"]),
                    str(r["final_full_test_acc"]), str(r["final_full_test_acc_train_mode"]),
                    str(r.get("mean_full_test_acc_last10", r.get("final_full_test_acc"))),
                    str(r.get("mean_task_acc_last10_2k", r.get("mean_task_acc_last10"))),
                    str(r["mean_adv_ssim_last10"]), str(r["mean_adv_psnr_last10"]),
                    str(r["final_adv_ssim_eval"]), str(r["final_adv_ssim_train"]), str(r["final_adv_ssim_max"]),
                    str(r["final_adv_psnr_max"]), str(r.get("final_dcor", "N/A")), str(r["training_time_s"]),
                    str(r["meets_utility_criterion"])
                ]
                f.write(",".join(row) + "\n")

    print(f"[HOÀN TẤT AR-TAPE v0] Toàn bộ kết quả đã được ghi vào {json_path} và {csv_path}")


if __name__ == "__main__":
    main()
