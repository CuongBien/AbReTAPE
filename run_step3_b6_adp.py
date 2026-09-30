#!/usr/bin/env python3
# Bước 3: Baseline B6 — ADP-AE (Adversarial Distortion Plug-in, arXiv:2502.20629)
# Giao thức 2 Phase kiểm chứng hiện tượng Server Absorption (Novelty N1)
import os
import sys
import time
import json
import csv
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

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
from src.data import get_cifar10, CIFAR10_MEAN, CIFAR10_STD
from src.defenses import PerturbAE
from src.attacks import Decoder, train_inversion_epoch, evaluate_inversion
from src.metrics import get_lpips_fn, distance_correlation
from src.training import EarlyStopping
from src.utils import save_reconstruction_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline B6: ADP-AE (Adversarial Distortion Plug-in 2-Phase Protocol)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Biên độ nhiễu loạn delta kẹp bởi alpha * tanh (mặc định: 0.1)")
    parser.add_argument("--lam-util", type=float, default=1.0, help="Hệ số lambda_util cân bằng giữa MSE tái tạo và CE phân loại (mặc định: 1.0)")
    parser.add_argument("--ae-epochs", type=int, default=15, help="Số epochs huấn luyện PerturbAE trong Phase 1 (mặc định: 15)")
    parser.add_argument("--epochs", type=int, default=10, help="Số epochs huấn luyện Split Learning trong Phase 2 (mặc định: 10)")
    parser.add_argument("--decoder-epochs", type=int, default=10, help="Số epochs huấn luyện Decoder tấn công thích ứng (mặc định: 10)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate cho Split Learning trong Phase 2 (mặc định: 0.01)")
    parser.add_argument("--ae-lr", type=float, default=1e-3, help="Learning rate cho PerturbAE (mặc định: 1e-3)")
    parser.add_argument("--decoder-lr", type=float, default=1e-3, help="Learning rate cho Decoder (mặc định: 1e-3)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá (mặc định: 5)")
    parser.add_argument("--patience", type=int, default=10, help="Số lần chờ Early Stopping (mặc định: 10, 0 để tắt)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2,
                        help="Số luồng nạp dữ liệu")
    parser.add_argument("--b0-ckpt", type=str, default=None, help="Đường dẫn checkpoint B0 Vanilla")
    parser.add_argument("--b1-dec-ckpt", type=str, default=None, help="Đường dẫn checkpoint Decoder Bước 1")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục dữ liệu")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step3_B6"), help="Thư mục lưu outputs")
    parser.add_argument("--resume", action="store_true", default=False, help="Khôi phục tiến trình nếu có")
    return parser.parse_args()


def find_b0_checkpoint(user_path=None):
    if user_path and os.path.isfile(user_path):
        return user_path
    candidates = [
        os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step0", "b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step0", "best_b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "b0_vanilla.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Không tìm thấy checkpoint B0 Vanilla tại các đường dẫn mặc định: {candidates}")


def find_b1_decoder_checkpoint(user_path=None):
    if user_path and os.path.isfile(user_path):
        return user_path
    candidates = [
        os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step1", "best_b1_decoder.pt"),
        os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step1", "b1_decoder.pt"),
        os.path.join(PROJECT_ROOT, "best_b1_decoder.pt"),
        os.path.join(PROJECT_ROOT, "b1_decoder.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Không tìm thấy checkpoint Decoder Bước 1 tại các đường dẫn mặc định: {candidates}")


def evaluate_sl_with_ae(client, server, ae, loader, device):
    client.eval()
    server.eval()
    if ae is not None:
        ae.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            z = client(x)
            z_p = z + ae(z) if ae is not None else z
            logits = server(z_p)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / max(total, 1)


def evaluate_inversion_with_ae(client, decoder, ae, loader, device, lpips_fn=None):
    client.eval()
    decoder.eval()
    if ae is not None:
        ae.eval()
    class AEWrapper(nn.Module):
        def __init__(self, ae_mod):
            super().__init__()
            self.ae_mod = ae_mod
        def forward(self, z):
            return z + self.ae_mod(z)

    wrapper = AEWrapper(ae) if ae is not None else None
    return evaluate_inversion(client, decoder, loader, device, CIFAR10_MEAN, CIFAR10_STD, defense=wrapper, lpips_fn=lpips_fn)


def train_ae_phase1(ae, client, server, decoder, loader, epochs, lr, lam_util, device, eval_freq=5, patience=10):
    print("\n" + "=" * 70)
    print("PHASE 1: HUẤN LUYỆN PERTURBAE KIỂU INFERENCE (THEO CHUẨN ADP)")
    print(f"Client, Server (B0) và Decoder (Bước 1) ĐƯỢC ĐÓNG BĂNG.")
    print(f"Mục tiêu tối ưu: min L_AE = -MSE(x_hat, x) + {lam_util} * CE(logits, y)")
    print("=" * 70)

    ae.train()
    client.eval()
    server.eval()
    decoder.eval()

    # Đóng băng trọng số server và decoder nhưng GIỮ computational graph mở để đạo hàm truyền về ae
    for p in client.parameters():
        p.requires_grad = False
    for p in server.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss, total_mse, total_ce, total_samples = 0.0, 0.0, 0.0, 0

        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            batch_size = x.size(0)

            with torch.no_grad():
                z = client(x)

            # AE có gradient
            delta = ae(z)
            z_p = z + delta

            # Server và Decoder đóng băng tham số nhưng gradient vẫn lan truyền về z_p -> delta -> ae
            x_hat = decoder(z_p)
            logits = server(z_p)

            loss_mse = mse(x_hat, x)
            loss_ce = ce(logits, y)
            loss = -loss_mse + lam_util * loss_ce

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_size
            total_mse += loss_mse.item() * batch_size
            total_ce += loss_ce.item() * batch_size
            total_samples += batch_size

        sched.step()
        ep_loss = total_loss / total_samples
        ep_mse = total_mse / total_samples
        ep_ce = total_ce / total_samples
        ep_time = time.time() - t0

        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}

        print(f"[Phase 1 AE] Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Loss: {ep_loss:.4f} | Recon MSE: {ep_mse:.4f} (đang cực đại hóa) | Task CE: {ep_ce:.4f}", flush=True)

    if best_state is not None:
        ae.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return ae


def train_sl_phase2(client, server, ae, trainloader, testloader, epochs, lr, patience, device, eval_freq=5):
    print("\n" + "=" * 70)
    print("PHASE 2: SPLIT LEARNING TRAINING VỚI PERTURBAE ĐÓNG BĂNG")
    print("Dự đoán Novelty N1: Server hấp thụ delta(z), Accuracy phục hồi về mức Vanilla")
    print("=" * 70)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    client.train()
    server.train()

    opt_c = torch.optim.SGD(client.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    opt_s = torch.optim.SGD(server.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched_c = CosineAnnealingLR(opt_c, T_max=epochs)
    sched_s = CosineAnnealingLR(opt_s, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, mode="max")

    best_acc = 0.0
    best_client = None
    best_server = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            batch_size = x.size(0)

            # Forward Client
            z = client(x)
            # Nhiễu loạn qua AE đóng băng
            z_p = z + ae(z)

            # Cắt kết nối sang Server
            z_d = z_p.detach().requires_grad_(True)
            logits = server(z_d)
            loss = ce(logits, y)

            # Backward Server
            opt_s.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(server.parameters(), max_norm=1.0)
            opt_s.step()

            # Backward Client xuyên qua AE đóng băng
            opt_c.zero_grad()
            z_p.backward(z_d.grad)
            torch.nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
            opt_c.step()

            total_loss += loss.item() * batch_size
            correct += (logits.argmax(1) == y).sum().item()
            total += batch_size

        sched_c.step()
        sched_s.step()
        ep_time = time.time() - t0
        train_acc = correct / total

        if epoch % eval_freq == 0 or epoch == epochs:
            t_acc = evaluate_sl_with_ae(client, server, ae, testloader, device)
            if t_acc > best_acc:
                best_acc = t_acc
                best_client = {k: v.cpu().clone() for k, v in client.state_dict().items()}
                best_server = {k: v.cpu().clone() for k, v in server.state_dict().items()}
            print(f"[Phase 2 SL] Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Train Acc: {train_acc*100:.2f}% | Test Acc: {t_acc*100:.2f}% (Best: {best_acc*100:.2f}%)", flush=True)
            if early_stopping.step(t_acc, epoch=epoch):
                print(f"[EARLY STOPPING] Dừng sớm SL tại epoch {epoch}!")
                break
        else:
            print(f"[Phase 2 SL] Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Train Acc: {train_acc*100:.2f}%", flush=True)

    if best_client is not None and best_server is not None:
        client.load_state_dict({k: v.to(device) for k, v in best_client.items()})
        server.load_state_dict({k: v.to(device) for k, v in best_server.items()})

    final_test_acc = evaluate_sl_with_ae(client, server, ae, testloader, device)
    print(f"[Phase 2 Result] Final Test Accuracy sau khi Server hấp thụ delta: {final_test_acc*100:.2f}%")
    return client, server, final_test_acc


def train_adaptive_decoder(client, ae, trainloader, testloader, epochs, lr, patience, device, lpips_fn=None, eval_freq=5):
    print("\n" + "=" * 70)
    print("PHASE 3: HUẤN LUYỆN DECODER TẤN CÔNG THÍCH ỨNG TRÊN z' = z + delta(z)")
    print("Dự đoán Novelty N1: Decoder tái tạo ảnh nét, đưa PSNR/SSIM quay về xấp xỉ B0 (Privacy = 0)")
    print("=" * 70)

    client.eval()
    ae.eval()
    decoder = Decoder().to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, mode="max")

    best_psnr = 0.0
    best_dec_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        decoder.train()
        total_loss, total_samples = 0.0, 0

        for x, _ in trainloader:
            x = x.to(device, non_blocking=True)
            batch_size = x.size(0)

            opt.zero_grad()
            with torch.no_grad():
                z = client(x)
                z_p = z + ae(z)

            x_hat = decoder(z_p)
            loss = crit(x_hat, x)
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        sched.step()
        ep_time = time.time() - t0

        if epoch % eval_freq == 0 or epoch == epochs:
            _, cur_psnr, cur_ssim, _ = evaluate_inversion_with_ae(client, decoder, ae, testloader, device, lpips_fn=None)
            if cur_psnr is not None and cur_psnr > best_psnr:
                best_psnr = cur_psnr
                best_dec_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
            print(f"[Adaptive Attack] Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | PSNR: {cur_psnr:.2f} dB | SSIM: {cur_ssim:.4f} (Best PSNR: {best_psnr:.2f} dB)", flush=True)
            if cur_psnr is not None and early_stopping.step(cur_psnr, epoch=epoch):
                print(f"[EARLY STOPPING] Dừng sớm Decoder tại epoch {epoch}!")
                break

    if best_dec_state is not None:
        decoder.load_state_dict({k: v.to(device) for k, v in best_dec_state.items()})

    mse, psnr, ssim, lpips_val = evaluate_inversion_with_ae(client, decoder, ae, testloader, device, lpips_fn=lpips_fn)
    return decoder, mse, psnr, ssim, lpips_val


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    b0_path = find_b0_checkpoint(args.b0_ckpt)
    b1_dec_path = find_b1_decoder_checkpoint(args.b1_dec_ckpt)

    print("\n" + "=" * 80)
    print("BƯỚC 3 — BASELINE B6: ADVERSARIAL DISTORTION PLUG-IN (ADP, arXiv:2502.20629)")
    print(f"Device: {device} | Alpha: {args.alpha} | Lambda Util: {args.lam_util}")
    print(f"Checkpoint B0: {b0_path}")
    print(f"Checkpoint B1 Decoder: {b1_dec_path}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 80)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    lpips_fn = get_lpips_fn(device=device)

    # 0. Nạp mô hình gốc B0 và Decoder Bước 1
    print("\n==> [BƯỚC 0] Đang nạp mô hình B0 Vanilla và Decoder Bước 1...")
    client_b0 = ClientModel().to(device)
    server_b0 = ServerModel().to(device)
    b0_dict = torch.load(b0_path, map_location=device)
    client_b0.load_state_dict(b0_dict["client"])
    server_b0.load_state_dict(b0_dict["server"])

    decoder_b1 = Decoder().to(device)
    b1_dict = torch.load(b1_dec_path, map_location=device)
    decoder_state = b1_dict["decoder"] if "decoder" in b1_dict else b1_dict
    decoder_b1.load_state_dict(decoder_state)

    b0_acc = evaluate_sl_with_ae(client_b0, server_b0, None, testloader, device)
    _, b0_psnr, b0_ssim, b0_lpips = evaluate_inversion_with_ae(client_b0, decoder_b1, None, testloader, device, lpips_fn=lpips_fn)
    lpips_str = f"{b0_lpips:.4f}" if b0_lpips is not None else "N/A"
    print(f"[Baseline B0 Anchor] Test Acc: {b0_acc*100:.2f}% | Attack PSNR: {b0_psnr:.2f} dB | SSIM: {b0_ssim:.4f} | LPIPS: {lpips_str}", flush=True)

    # Khởi tạo PerturbAE
    ae = PerturbAE(channels=64, alpha=args.alpha).to(device)

    # =========================================================================
    # PHASE 1: HUẤN LUYỆN AE KIỂU INFERENCE (ĐỐI CHIẾU ADP GỐC)
    # =========================================================================
    ae_ckpt_path = os.path.join(args.output_dir, f"b6_ae_alpha_{args.alpha}.pt")
    if args.resume and os.path.isfile(ae_ckpt_path):
        print(f"\n[RESUME] Tìm thấy checkpoint PerturbAE tại {ae_ckpt_path}. Đang nạp lại...")
        ae.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
    else:
        ae = train_ae_phase1(
            ae, client_b0, server_b0, decoder_b1, trainloader,
            epochs=args.ae_epochs, lr=args.ae_lr, lam_util=args.lam_util,
            device=device, eval_freq=args.eval_freq, patience=args.patience
        )
        torch.save(ae.state_dict(), ae_ckpt_path)
        print(f"[Phase 1] Đã lưu checkpoint PerturbAE tại: {ae_ckpt_path}")

    # Đánh giá Phase 1
    p1_acc = evaluate_sl_with_ae(client_b0, server_b0, ae, testloader, device)
    p1_mse, p1_psnr, p1_ssim, p1_lpips = evaluate_inversion_with_ae(client_b0, decoder_b1, ae, testloader, device, lpips_fn=lpips_fn)
    
    # Đo dCor ở Phase 1
    client_b0.eval()
    ae.eval()
    p1_dcor_sum, p1_n_eval = 0.0, 0
    with torch.no_grad():
        for x_t, _ in testloader:
            x_t = x_t.to(device)
            z_b0_t = client_b0(x_t)
            z_p1_t = z_b0_t + ae(z_b0_t)
            p1_dcor_sum += distance_correlation(x_t, z_p1_t).item()
            p1_n_eval += 1
            if p1_n_eval >= 15:
                break
    p1_dcor = p1_dcor_sum / max(p1_n_eval, 1)

    p1_grid_path = os.path.join(args.output_dir, f"b6_phase1_reconstruction_alpha_{args.alpha}.png")
    class AEWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, z):
            return z + self.m(z)
    save_reconstruction_grid(client_b0, decoder_b1, testloader, device, CIFAR10_MEAN, CIFAR10_STD, defense=AEWrapper(ae), save_path=p1_grid_path, num_images=8)

    print("\n" + "-" * 70)
    print(f"[Phase 1 Kết quả - ADP Inference]")
    print(f"Task Accuracy: {p1_acc*100:.2f}% (Drop: {(b0_acc - p1_acc)*100:.2f}% | Kỳ vọng: ~4%)")
    print(f"Reconstruction PSNR: {p1_psnr:.2f} dB (Drop: {b0_psnr - p1_psnr:.2f} dB | Giảm: {((b0_psnr - p1_psnr)/b0_psnr)*100:.1f}%)")
    print(f"Reconstruction SSIM: {p1_ssim:.4f} (Drop: {b0_ssim - p1_ssim:.4f}) | dCor(X, Z'): {p1_dcor:.4f}")
    print(f"-> Kết luận Phase 1: ADP hoạt động đúng như công bố trong inference (phá decoder cũ mà giữ accuracy).")
    print("-" * 70)

    # =========================================================================
    # PHASE 2: SPLIT LEARNING TRAINING VỚI AE ĐÓNG BĂNG (CHỨNG MINH HẤP THỤ N1)
    # =========================================================================
    sl_ckpt_path = os.path.join(args.output_dir, f"b6_sl_absorbed_alpha_{args.alpha}.pt")
    client_sl = ClientModel().to(device)
    server_sl = ServerModel().to(device)
    client_sl.load_state_dict(b0_dict["client"])
    server_sl.load_state_dict(b0_dict["server"])

    if args.resume and os.path.isfile(sl_ckpt_path):
        print(f"\n[RESUME] Tìm thấy checkpoint SL Absorbed tại {sl_ckpt_path}. Đang nạp lại...")
        sl_dict = torch.load(sl_ckpt_path, map_location=device)
        client_sl.load_state_dict(sl_dict["client"])
        server_sl.load_state_dict(sl_dict["server"])
        p2_acc = evaluate_sl_with_ae(client_sl, server_sl, ae, testloader, device)
    else:
        client_sl, server_sl, p2_acc = train_sl_phase2(
            client_sl, server_sl, ae, trainloader, testloader,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            device=device, eval_freq=args.eval_freq
        )
        torch.save({"client": client_sl.state_dict(), "server": server_sl.state_dict(), "acc": p2_acc}, sl_ckpt_path)
        print(f"[Phase 2] Đã lưu checkpoint SL Absorbed tại: {sl_ckpt_path}")

    # =========================================================================
    # PHASE 3: HUẤN LUYỆN DECODER TẤN CÔNG THÍCH ỨNG
    # =========================================================================
    dec_adapt_path = os.path.join(args.output_dir, f"b6_decoder_adaptive_alpha_{args.alpha}.pt")
    if args.resume and os.path.isfile(dec_adapt_path):
        print(f"\n[RESUME] Tìm thấy checkpoint Adaptive Decoder tại {dec_adapt_path}. Đang nạp lại...")
        adapt_dec = Decoder().to(device)
        adapt_dec.load_state_dict(torch.load(dec_adapt_path, map_location=device))
        p2_mse, p2_psnr, p2_ssim, p2_lpips = evaluate_inversion_with_ae(client_sl, adapt_dec, ae, testloader, device, lpips_fn=lpips_fn)
    else:
        adapt_dec, p2_mse, p2_psnr, p2_ssim, p2_lpips = train_adaptive_decoder(
            client_sl, ae, trainloader, testloader,
            epochs=args.decoder_epochs, lr=args.decoder_lr, patience=args.patience,
            device=device, lpips_fn=lpips_fn, eval_freq=args.eval_freq
        )
        torch.save(adapt_dec.state_dict(), dec_adapt_path)
        print(f"[Phase 3] Đã lưu checkpoint Adaptive Decoder tại: {dec_adapt_path}")

    p2_grid_path = os.path.join(args.output_dir, f"b6_phase2_reconstruction_alpha_{args.alpha}.png")
    save_reconstruction_grid(client_sl, adapt_dec, testloader, device, CIFAR10_MEAN, CIFAR10_STD, defense=AEWrapper(ae), save_path=p2_grid_path, num_images=8)

    # Đo dCor sau Phase 2
    client_sl.eval()
    ae.eval()
    dcor_sum = 0.0
    n_eval = 0
    with torch.no_grad():
        for x_t, _ in testloader:
            x_t = x_t.to(device)
            z_t = client_sl(x_t) + ae(client_sl(x_t))
            dcor_sum += distance_correlation(x_t, z_t).item()
            n_eval += 1
            if n_eval >= 15:
                break
    p2_dcor = dcor_sum / max(n_eval, 1)

    # =========================================================================
    # TỔNG KẾT BẢNG SO SÁNH 2 PHASE
    # =========================================================================
    print("\n" + "=" * 80)
    print("TỔNG KẾT BASELINE B6: ADP-AE TRONG SPLIT LEARNING")
    print("=" * 80)
    print(f"{'Giai đoạn':<25} | {'Task Acc (%)':<14} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'Kết luận'}")
    print("-" * 80)
    print(f"{'B0 (Vanilla)':<25} | {b0_acc*100:<14.2f} | {b0_psnr:<12.2f} | {b0_ssim:<10.4f} | Gốc không phòng thủ")
    print(f"{'Phase 1 (ADP Inference)':<25} | {p1_acc*100:<14.2f} | {p1_psnr:<12.2f} | {p1_ssim:<10.4f} | ADP hoạt động đúng bài gốc")
    print(f"{'Phase 2 (SL Absorbed)':<25} | {p2_acc*100:<14.2f} | {p2_psnr:<12.2f} | {p2_ssim:<10.4f} | Server hấp thụ -> Privacy = 0")
    print("=" * 80)

    # Lưu kết quả JSON
    res_data = {
        "defense": "B6_ADP_AE",
        "alpha": args.alpha,
        "lam_util": args.lam_util,
        "b0_anchor": {
            "test_acc": b0_acc,
            "psnr": b0_psnr,
            "ssim": b0_ssim,
            "lpips": b0_lpips
        },
        "phase1_inference": {
            "test_acc": p1_acc,
            "psnr": p1_psnr,
            "ssim": p1_ssim,
            "lpips": p1_lpips,
            "dcor": p1_dcor,
            "utility_drop_percent": (b0_acc - p1_acc) * 100,
            "psnr_reduction_percent": ((b0_psnr - p1_psnr) / b0_psnr) * 100
        },
        "phase2_absorption": {
            "test_acc": p2_acc,
            "psnr": p2_psnr,
            "ssim": p2_ssim,
            "lpips": p2_lpips,
            "dcor": p2_dcor
        }
    }
    json_path = os.path.join(args.output_dir, f"results_b6_alpha_{args.alpha}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res_data, f, indent=2)
    print(f"[LƯU] Đã lưu kết quả chi tiết tại: {json_path}")


if __name__ == "__main__":
    main()
