#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_pcam_b7_fsha.py
==============================================================================
ĐÁNH GIÁ TẤN CÔNG CHỦ ĐỘNG FSHA (FEATURE SPACE HIJACKING ATTACK)
TRÊN CLIENT B7 LIGHTSPLIT (k=512, MODE F) TRÊN PATCHCAMELYON (PCAM)
==============================================================================
Giao thức chuẩn hóa hoàn toàn tương đồng với AR-TAPE:
- Phân chia 25.000 ảnh Private (Client) và 25.000 ảnh Public (Pilot AE).
- Pilot AutoEncoder (latent_channels=64, upscale_factor=2) huấn luyện 10 epochs.
- FSHA Discriminator (in_channels=64, Pixel + Spatial multi-scale).
- Hai kịch bản tấn công: grad_scale = 1.0 và grad_scale = 5.0 (30 epochs mỗi kịch bản).
- Client và Server được nạp từ checkpoint tối ưu nhất (Best Validation AUC).
- Đánh giá tại epoch 30: Test SSIM, Test PSNR (dB), Test MSE, LPIPS và Converged Task Accuracy.
==============================================================================
"""

import os
import sys
import time
import json
import csv
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from src.models import (
    ClientWithBottleneck,
    ServerWithBottleneck
)
from src.attacks.fsha import (
    FSHADiscriminator,
    FSHAPilotAutoEncoder,
    fit_fsha_pilot
)
from src.data.medical import get_pcam_dataloaders
from src.metrics.clinical import compute_binary_clinical_metrics
from src.metrics.reconstruction import psnr_ssim, calculate_lpips, get_lpips_fn, denormalize


class TeeLogger:
    """Tự động ghi đồng thời ra màn hình terminal và file log."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        try:
            self.log_file.write(message)
            self.log_file.flush()
        except Exception:
            pass

    def flush(self):
        self.terminal.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass


def get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "2afad93"


class FSHADecoderWrapper(nn.Module):
    """
    Wrapper kết nối Pilot Decoder trong FSHA để giải mã từ zt (B, k):
    zt (B, k) -> Server Lift (R^T) -> zh (B, 64, 32, 32) -> Pilot Decoder -> x_rec (B, 3, 64, 64)
    """
    def __init__(self, pilot_ae, server):
        super().__init__()
        self.pilot_ae = pilot_ae
        self.server = server

    def forward(self, zt):
        zh = self.server.lift(zt)
        zh_tensor = zh.view(zt.size(0), 64, 32, 32)
        return self.pilot_ae.decode(zh_tensor)


@torch.no_grad()
def evaluate_split(client, server, data_loader, criterion, device, max_batches=None):
    client.eval()
    server.eval()

    total_loss = 0.0
    all_targets = []
    all_probs = []

    for idx, (x, y) in enumerate(data_loader):
        if max_batches is not None and idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        zt = client(x)
        logits = server(zt)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy().tolist())
        all_targets.extend(y.cpu().numpy().tolist())

    n_samples = len(all_targets)
    mean_loss = total_loss / max(1, n_samples)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    accuracy = (sum(1 for p, t in zip(preds, all_targets) if p == t) / max(1, n_samples)) * 100.0

    clin_metrics = compute_binary_clinical_metrics(all_targets, all_probs)
    auc_roc = clin_metrics.get("auc_roc", 0.5)
    return mean_loss, accuracy, auc_roc, clin_metrics


@torch.no_grad()
def evaluate_reconstruction_attacker(client, attacker, data_loader, device, mean, std, lpips_fn=None, max_batches=None):
    client.eval()
    attacker.eval()

    total_samples = 0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_mse = 0.0
    criterion = nn.MSELoss()

    for idx, (x, _) in enumerate(data_loader):
        if max_batches is not None and idx >= max_batches:
            break
        x = x.to(device)
        zt = client(x)
        x_rec = attacker(zt)

        mse_val = criterion(x_rec, x).item()
        total_mse += mse_val * x.size(0)

        p, s = psnr_ssim(x_rec, x, mean=mean, std=std)
        total_psnr += p * x.size(0)
        total_ssim += s * x.size(0)

        if lpips_fn is not None:
            lp = calculate_lpips(x_rec, x, mean=mean, std=std, lpips_fn=lpips_fn)
            total_lpips += lp * x.size(0)

        total_samples += x.size(0)

    return {
        "mse": float(total_mse / max(1, total_samples)),
        "psnr": float(total_psnr / max(1, total_samples)),
        "ssim": float(total_ssim / max(1, total_samples)),
        "lpips": float(total_lpips / max(1, total_samples)) if lpips_fn is not None else None,
        "num_samples": total_samples
    }


def save_reconstruction_sample_grid(client, attacker, data_loader, device, mean, std, save_path, num_images=16):
    client.eval()
    attacker.eval()

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            zt = client(x)
            x_rec = attacker(zt)
            break

    n = min(num_images, x.size(0))
    x = x[:n].cpu()
    x_rec = x_rec[:n].cpu()

    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    x_orig = torch.clamp(x * s + m, 0.0, 1.0).numpy()
    x_recon = torch.clamp(x_rec * s + m, 0.0, 1.0).numpy()

    cols = min(8, n)
    rows = int(np.ceil(n / cols)) * 2

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)

    for i in range(n):
        r_orig = (i // cols) * 2
        r_rec = r_orig + 1
        c = i % cols

        img_orig = np.transpose(x_orig[i], (1, 2, 0))
        img_rec = np.transpose(x_recon[i], (1, 2, 0))

        axes[r_orig, c].imshow(img_orig)
        axes[r_orig, c].set_title(f"Gốc #{i+1}", fontsize=9)
        axes[r_orig, c].axis("off")

        axes[r_rec, c].imshow(img_rec)
        axes[r_rec, c].set_title(f"Tái tạo #{i+1}", fontsize=9)
        axes[r_rec, c].axis("off")

    for r in range(rows):
        for c in range(cols):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[TRỰC QUAN HÓA] Đã lưu lưới ảnh đối chứng tại: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá FSHA trên B7 LightSplit (k=512, Mode F) trên PCAM")
    parser.add_argument("--data-dir", type=str, default="./data/pcam_processed", help="Thư mục npy PCAM")
    parser.add_argument("--b7-dir", type=str, default="./output/pcam_b7_k512", help="Thư mục chứa kết quả B7 k=512")
    parser.add_argument("--k", type=int, default=512, help="Số chiều bottleneck (mặc định: 512)")
    parser.add_argument("--m", type=int, default=512, help="Số chiều ẩn MLP (mặc định: 512)")
    parser.add_argument("--D", type=int, default=65536, help="Số chiều cut-layer (mặc định: 65536)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024], help="Danh sách seeds cần đánh giá")
    parser.add_argument("--fsha-epochs", type=int, default=30, help="Số epochs huấn luyện FSHA (mặc định: 30)")
    parser.add_argument("--grad-scales", nargs="+", type=float, default=[1.0, 5.0], help="Danh sách grad_scale FSHA")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Nạp checkpoint có sẵn nếu có")
    parser.add_argument("--dry-run", action="store_true", help="Chạy thử 1 batch")
    return parser.parse_args()


def run_fsha_for_seed(args, seed, train_loader, test_loader, meta, device):
    seed_output_dir = os.path.join(args.b7_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)
    ckpt_dir = os.path.join(seed_output_dir, "checkpoints")

    best_ckpt_p = os.path.join(ckpt_dir, "best_checkpoint.pt")
    best_client_p = os.path.join(ckpt_dir, "best_auc_b7_client.pt")
    best_server_p = os.path.join(ckpt_dir, "best_auc_b7_server.pt")

    log_file_p = os.path.join(seed_output_dir, f"eval_pcam_b7_fsha_seed_{seed}.log")
    tee = TeeLogger(log_file_p)
    sys.stdout = tee
    sys.stderr = tee

    mean, std = meta["mean"], meta["std"]
    max_b = 2 if args.dry_run else None
    lpips_fn = get_lpips_fn(device=device)

    print("\n" + "=" * 90)
    print(f"BẮT ĐẦU ĐÁNH GIÁ FSHA TRÊN CLIENT B7 LIGHTSPLIT (k={args.k}) | SEED {seed}")
    print(f"Thiết bị: {device} | Output Dir: {seed_output_dir}")
    print("=" * 90, flush=True)

    # 1. Khởi tạo Client và Server B7
    client = ClientWithBottleneck(
        k=args.k,
        D=args.D,
        init_ortho=True,
        seed=seed,
        fixed_bottleneck=True,
        in_channels=3,
        stem_stride=2
    ).to(device)

    server = ServerWithBottleneck(
        k=args.k,
        m=args.m,
        D=args.D,
        num_classes=2,
        unprojector_mode="linear_fixed",
        ortho_matrix=client.R,
        seed=seed
    ).to(device)

    # Nạp trọng số sạch tại best validation epoch
    if os.path.isfile(best_client_p):
        print(f"[NẠP TRỌNG SỐ] Nạp Client từ: {best_client_p}")
        client.load_state_dict(torch.load(best_client_p, map_location=device, weights_only=False))
    elif os.path.isfile(best_ckpt_p):
        print(f"[NẠP TRỌNG SỐ] Nạp Client từ best_checkpoint.pt: {best_ckpt_p}")
        ckpt_b = torch.load(best_ckpt_p, map_location=device, weights_only=False)
        client.load_state_dict(ckpt_b["client_state_dict"])
    else:
        raise FileNotFoundError(f"Không tìm thấy checkpoint B7 cho Seed {seed} tại {ckpt_dir}!")

    if os.path.isfile(best_server_p):
        print(f"[NẠP TRỌNG SỐ] Nạp Server từ: {best_server_p}")
        server.load_state_dict(torch.load(best_server_p, map_location=device, weights_only=False))
    elif os.path.isfile(best_ckpt_p):
        ckpt_b = torch.load(best_ckpt_p, map_location=device, weights_only=False)
        server.load_state_dict(ckpt_b["server_state_dict"])

    # Lưu bản sao sạch vào RAM để reset trước mỗi kịch bản tấn công
    clean_client_sd = {k: v.clone() for k, v in client.state_dict().items()}
    clean_server_sd = {k: v.clone() for k, v in server.state_dict().items()}

    criterion_eval = nn.CrossEntropyLoss()
    base_loss, base_acc, base_auc, _ = evaluate_split(client, server, test_loader, criterion_eval, device, max_batches=max_b)
    print(f"[TEST GỐC] Clean Model - Test AUC: {base_auc:.4f} | Test Acc: {base_acc:.2f}%\n")

    # 2. Phân chia dữ liệu: 25k private train và 25k public train (cùng giao thức AR-TAPE)
    train_ds_all = train_loader.dataset
    n_all_train = len(train_ds_all)
    n_priv = n_all_train // 2
    g_split = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_all_train, generator=g_split).tolist()
    priv_set = Subset(train_ds_all, perm[:n_priv])
    pub_set = Subset(train_ds_all, perm[n_priv:])

    use_cuda = torch.cuda.is_available()
    fsha_priv_loader = DataLoader(priv_set, batch_size=args.batch_size, shuffle=True, pin_memory=use_cuda)
    fsha_pub_loader = DataLoader(pub_set, batch_size=args.batch_size, shuffle=True, pin_memory=use_cuda)

    # 3. Pilot AutoEncoder (10 epochs trên 25k Public)
    pilot_ae = FSHAPilotAutoEncoder(
        in_channels=3,
        latent_channels=64,
        stem_stride=2,
        upscale_factor=2
    ).to(device)

    pilot_ckpt_p = os.path.join(seed_output_dir, f"pcam_b7_pilot_ae_seed_{seed}.pt")
    # Kiểm tra xem có thể dùng chung checkpoint pilot_ae từ artape nếu có
    artape_pilot_p = os.path.join(os.path.dirname(args.b7_dir), "pcam_m4_artape", f"seed_{seed}", f"pcam_artape_pilot_ae_seed_{seed}.pt")
    if args.resume and os.path.isfile(pilot_ckpt_p):
        print(f"[RESUME] Đã tìm thấy checkpoint Pilot AutoEncoder tại: {pilot_ckpt_p}. Nạp trọng số...")
        pilot_ae.load_state_dict(torch.load(pilot_ckpt_p, map_location=device, weights_only=False))
    elif args.resume and os.path.isfile(artape_pilot_p):
        print(f"[RESUME] Dùng lại Pilot AutoEncoder đã khớp từ AR-TAPE: {artape_pilot_p}")
        pilot_ae.load_state_dict(torch.load(artape_pilot_p, map_location=device, weights_only=False))
        torch.save(pilot_ae.state_dict(), pilot_ckpt_p)
    else:
        print("[FSHA] Huấn luyện Pilot AutoEncoder trên 25k ảnh Public (10 epochs)...")
        fit_fsha_pilot(
            pilot_ae=pilot_ae,
            pilot_server=None,
            pub_loader=fsha_pub_loader,
            test_loader=test_loader,
            epochs=1 if args.dry_run else 10,
            lr=1e-3,
            device=device,
            mean=mean,
            std=std
        )
        torch.save(pilot_ae.state_dict(), pilot_ckpt_p)
        print(f"[FSHA] Pilot AutoEncoder đã hội tụ và lưu tại: {pilot_ckpt_p}")

    total_atk_epochs = 1 if args.dry_run else args.fsha_epochs
    fsha_results = {}

    for gs in args.grad_scales:
        fsha_name = f"fsha_gs{gs}"
        fsha_desc = f"FSHA Active Hijacking Attack on B7 k={args.k} (grad_scale={gs}, {total_atk_epochs} epochs)"
        print("\n" + "-" * 80)
        print(f"Bắt đầu: {fsha_name.upper()} - {fsha_desc}")
        print("-" * 80)

        # Khôi phục Client và Server về trạng thái sạch gốc
        client.load_state_dict({k: v.clone() for k, v in clean_client_sd.items()})
        server.load_state_dict({k: v.clone() for k, v in clean_server_sd.items()})

        fsha_ckpt_p = os.path.join(seed_output_dir, f"pcam_b7_{fsha_name}_seed_{seed}.pt")
        discriminator = FSHADiscriminator(in_channels=64).to(device)
        fsha_decoder = FSHADecoderWrapper(pilot_ae=pilot_ae, server=server).to(device)

        if args.resume and os.path.isfile(fsha_ckpt_p):
            print(f"[RESUME] Đã tìm thấy checkpoint FSHA tại: {fsha_ckpt_p}. Nạp và đánh giá...")
            fsha_ckpt_data = torch.load(fsha_ckpt_p, map_location=device, weights_only=False)
            if "client" in fsha_ckpt_data:
                client.load_state_dict(fsha_ckpt_data["client"])
            if "server" in fsha_ckpt_data:
                server.load_state_dict(fsha_ckpt_data["server"])
            if "discriminator" in fsha_ckpt_data:
                discriminator.load_state_dict(fsha_ckpt_data["discriminator"])
            fsha_dur = 0.0
            fsha_hist = []
        else:
            client.train()
            for p in client.parameters():
                p.requires_grad = True

            server.train()
            for p in server.parameters():
                p.requires_grad = True

            opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
            opt_c = torch.optim.SGD(client.parameters(), lr=1e-4, momentum=0.9)
            opt_s = torch.optim.SGD(server.parameters(), lr=1e-3, momentum=0.9)
            criterion_task_fsha = nn.CrossEntropyLoss()

            t_fsha0 = time.time()
            fsha_hist = []

            for f_ep in range(1, total_atk_epochs + 1):
                discriminator.train()
                client.train()
                server.train()

                d_loss_acc = 0.0
                n_f_samples = 0

                pub_iter = iter(fsha_pub_loader)
                for b_idx, (x_priv, y_priv) in enumerate(fsha_priv_loader):
                    if max_b is not None and b_idx >= max_b:
                        break
                    try:
                        x_pub, _ = next(pub_iter)
                    except StopIteration:
                        pub_iter = iter(fsha_pub_loader)
                        x_pub, _ = next(pub_iter)

                    x_priv, y_priv = x_priv.to(device), y_priv.to(device)
                    x_pub = x_pub.to(device)

                    # 1. Update Discriminator: z_pilot vs z_client (sau khi nâng chiều R^T về 64x32x32)
                    opt_d.zero_grad()
                    with torch.no_grad():
                        zt_priv = client(x_priv)
                        zh_priv = server.lift(zt_priv).view(zt_priv.size(0), 64, 32, 32)
                        z_pub, _ = pilot_ae(x_pub)

                    d_real = discriminator(z_pub)
                    d_fake = discriminator(zh_priv)

                    loss_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
                    loss_d.backward()
                    opt_d.step()

                    # 2. Adversarial Gradient Hijacking vào Client & thích ứng Server
                    opt_c.zero_grad()
                    opt_s.zero_grad()

                    zt_priv = client(x_priv)
                    zh_priv = server.lift(zt_priv).view(zt_priv.size(0), 64, 32, 32)

                    logits = server.base_server(zh_priv)
                    loss_task = criterion_task_fsha(logits, y_priv)

                    d_fake_adv = discriminator(zh_priv)
                    loss_adv = - d_fake_adv.mean()

                    loss_hijack = loss_task + gs * loss_adv
                    loss_hijack.backward()

                    opt_c.step()
                    opt_s.step()

                    d_loss_acc += loss_d.item() * x_priv.size(0)
                    n_f_samples += x_priv.size(0)

                ep_d_loss = d_loss_acc / max(1, n_f_samples)

                if f_ep == 1 or f_ep % 5 == 0 or f_ep == total_atk_epochs:
                    client.eval()
                    server.eval()
                    fsha_decoder.eval()
                    pilot_ae.eval()

                    fsha_eval = evaluate_reconstruction_attacker(
                        client, fsha_decoder, test_loader, device, mean, std, lpips_fn=None, max_batches=max_b
                    )
                    _, fsha_task_acc, fsha_task_auc, _ = evaluate_split(client, server, test_loader, criterion_eval, device, max_batches=max_b)

                    print(
                        f"  [FSHA Ep {f_ep:2d}/{total_atk_epochs:2d}] D Loss: {ep_d_loss:.4f} | "
                        f"Task Acc: {fsha_task_acc:5.2f}% | Test PSNR: {fsha_eval['psnr']:5.2f} dB | Test SSIM: {fsha_eval['ssim']:.4f}",
                        flush=True
                    )
                    fsha_hist.append({
                        "epoch": f_ep,
                        "d_loss": ep_d_loss,
                        "task_acc": fsha_task_acc,
                        "task_auc": fsha_task_auc,
                        "test_psnr": fsha_eval["psnr"],
                        "test_ssim": fsha_eval["ssim"]
                    })

            fsha_dur = time.time() - t_fsha0
            print(f"Huấn luyện {fsha_name.upper()} hoàn tất trong {fsha_dur/60:.1f} phút ({fsha_dur:.1f}s).")
            torch.save({
                "client": client.state_dict(),
                "server": server.state_dict(),
                "discriminator": discriminator.state_dict()
            }, fsha_ckpt_p)
            print(f"Đã lưu checkpoint {fsha_name.upper()} tại: {fsha_ckpt_p}")

        # Đánh giá chính thức FSHA tại epoch cuối 30
        client.eval()
        server.eval()
        fsha_decoder.eval()
        pilot_ae.eval()
        final_fsha_eval = evaluate_reconstruction_attacker(
            client, fsha_decoder, test_loader, device, mean, std, lpips_fn=lpips_fn, max_batches=max_b
        )
        _, final_fsha_acc, final_fsha_auc, _ = evaluate_split(client, server, test_loader, criterion_eval, device, max_batches=max_b)

        grid_p = os.path.join(seed_output_dir, f"recon_pcam_b7_{fsha_name}_seed_{seed}.png")
        save_reconstruction_sample_grid(client, fsha_decoder, test_loader, device, mean, std, grid_p, num_images=16)

        fsha_results[fsha_name] = {
            "attacker_type": fsha_name,
            "grad_scale": gs,
            "description": fsha_desc,
            "final_epoch": total_atk_epochs,
            "converged_task_acc": final_fsha_acc,
            "converged_task_auc": final_fsha_auc,
            "test_ssim": final_fsha_eval["ssim"],
            "test_psnr": final_fsha_eval["psnr"],
            "test_mse": final_fsha_eval["mse"],
            "test_lpips": final_fsha_eval["lpips"],
            "training_time_s": fsha_dur,
            "history": fsha_hist
        }

    # Xuất file kết quả FSHA riêng và cập nhật file results_pcam_b7_seed_{seed}.json gốc
    fsha_json_p = os.path.join(seed_output_dir, f"results_pcam_b7_fsha_seed_{seed}.json")
    with open(fsha_json_p, "w", encoding="utf-8") as f:
        json.dump(fsha_results, f, indent=2)

    b7_orig_json = os.path.join(seed_output_dir, f"results_pcam_b7_seed_{seed}.json")
    if os.path.isfile(b7_orig_json):
        try:
            with open(b7_orig_json, "r", encoding="utf-8") as f:
                orig_data = json.load(f)
            if "attackers" not in orig_data:
                orig_data["attackers"] = {}
            for k_f, v_f in fsha_results.items():
                orig_data["attackers"][k_f] = v_f
            if "summary_for_poc" in orig_data:
                orig_data["summary_for_poc"]["fsha_gs1_ssim"] = round(fsha_results.get("fsha_gs1.0", {}).get("test_ssim", 0.0), 4)
                orig_data["summary_for_poc"]["fsha_gs1_psnr"] = round(fsha_results.get("fsha_gs1.0", {}).get("test_psnr", 0.0), 2)
                orig_data["summary_for_poc"]["fsha_gs1_task_acc"] = round(fsha_results.get("fsha_gs1.0", {}).get("converged_task_acc", 0.0), 2)
                orig_data["summary_for_poc"]["fsha_gs5_ssim"] = round(fsha_results.get("fsha_gs5.0", {}).get("test_ssim", 0.0), 4)
                orig_data["summary_for_poc"]["fsha_gs5_psnr"] = round(fsha_results.get("fsha_gs5.0", {}).get("test_psnr", 0.0), 2)
                orig_data["summary_for_poc"]["fsha_gs5_task_acc"] = round(fsha_results.get("fsha_gs5.0", {}).get("converged_task_acc", 0.0), 2)
            with open(b7_orig_json, "w", encoding="utf-8") as f:
                json.dump(orig_data, f, indent=2)
            print(f"[CẬP NHẬT] Đã cập nhật metrics FSHA vào: {b7_orig_json}")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể cập nhật vào file gốc: {e}")

    print("\n" + "=" * 90)
    print(f"TỔNG KẾT FSHA TRÊN CLIENT B7 (k={args.k}) | SEED {seed}:")
    print(f"  * FSHA (gs=1.0): SSIM = {fsha_results['fsha_gs1.0']['test_ssim']:.4f} | PSNR = {fsha_results['fsha_gs1.0']['test_psnr']:.2f} dB | Task Acc = {fsha_results['fsha_gs1.0']['converged_task_acc']:.2f}%")
    print(f"  * FSHA (gs=5.0): SSIM = {fsha_results['fsha_gs5.0']['test_ssim']:.4f} | PSNR = {fsha_results['fsha_gs5.0']['test_psnr']:.2f} dB | Task Acc = {fsha_results['fsha_gs5.0']['converged_task_acc']:.2f}%")
    print("=" * 90 + "\n", flush=True)

    tee.close()
    return fsha_results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_p = os.path.join(args.data_dir, "pcam_stats.json")
    stats_dict = None
    if os.path.isfile(stats_p):
        with open(stats_p, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)

    train_loader, _, test_loader, meta = get_pcam_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        stats=stats_dict
    )

    all_seed_results = {}
    for seed in args.seeds:
        res = run_fsha_for_seed(args, seed, train_loader, test_loader, meta, device)
        all_seed_results[seed] = res

    # Thống kê tổng hợp đa seed
    print("\n" + "=" * 100)
    print(f"BÁO CÁO TỔNG HỢP FSHA TRÊN B7 LIGHTSPLIT (k={args.k}) - {len(args.seeds)} SEEDS")
    print("=" * 100)
    print("| Seed | FSHA gs=1 SSIM | FSHA gs=1 PSNR (dB) | FSHA gs=1 Acc (%) | FSHA gs=5 SSIM | FSHA gs=5 PSNR (dB) | FSHA gs=5 Acc (%) |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    gs1_ssims = [all_seed_results[s]["fsha_gs1.0"]["test_ssim"] for s in args.seeds]
    gs1_psnrs = [all_seed_results[s]["fsha_gs1.0"]["test_psnr"] for s in args.seeds]
    gs1_accs = [all_seed_results[s]["fsha_gs1.0"]["converged_task_acc"] for s in args.seeds]

    gs5_ssims = [all_seed_results[s]["fsha_gs5.0"]["test_ssim"] for s in args.seeds]
    gs5_psnrs = [all_seed_results[s]["fsha_gs5.0"]["test_psnr"] for s in args.seeds]
    gs5_accs = [all_seed_results[s]["fsha_gs5.0"]["converged_task_acc"] for s in args.seeds]

    for s in args.seeds:
        r = all_seed_results[s]
        print(
            f"| `{s}` | **{r['fsha_gs1.0']['test_ssim']:.4f}** | {r['fsha_gs1.0']['test_psnr']:.2f} dB | {r['fsha_gs1.0']['converged_task_acc']:.2f}% | "
            f"**{r['fsha_gs5.0']['test_ssim']:.4f}** | {r['fsha_gs5.0']['test_psnr']:.2f} dB | {r['fsha_gs5.0']['converged_task_acc']:.2f}% |"
        )

    def stats(vals):
        a = np.array(vals, dtype=float)
        return float(np.mean(a)), (float(np.std(a, ddof=1)) if len(a) > 1 else 0.0)

    m_s1, sd_s1 = stats(gs1_ssims)
    m_p1, sd_p1 = stats(gs1_psnrs)
    m_a1, sd_a1 = stats(gs1_accs)
    m_s5, sd_s5 = stats(gs5_ssims)
    m_p5, sd_p5 = stats(gs5_psnrs)
    m_a5, sd_a5 = stats(gs5_accs)

    print(
        f"| **Mean ± Std (n={len(args.seeds)})** | "
        f"**{m_s1:.4f} ± {sd_s1:.4f}** | **{m_p1:.2f} ± {sd_p1:.2f}** | **{m_a1:.2f} ± {sd_a1:.2f}%** | "
        f"**{m_s5:.4f} ± {sd_s5:.4f}** | **{m_p5:.2f} ± {sd_p5:.2f}** | **{m_a5:.2f} ± {sd_a5:.2f}%** |"
    )
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
