#!/usr/bin/env python3
# Bước 4: Triển khai & Thực nghiệm Tấn công Chủ động FSHA (Feature Space Hijacking Attack - Pasquini et al., CCS 2021)
# Đánh giá toàn diện 4 mũi đo M1 (B0), M2 (B1, B2), M3 (B3 NoPeek), M4 (B4, B5, B6) và đối chứng Novelty N1 & CV2.
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

from src.models.resnet_split import ClientModel, ServerModel
from src.data import get_cifar10_fsha_splits, CIFAR10_MEAN, CIFAR10_STD
from src.defenses import (
    GaussianNoise,
    DPSGDClientOptimizer,
    compute_dp_epsilon,
    NoPeekDefense,
    BlockScrambleDefense,
    DeformableOperatorDefense,
    ADPAutoEncoderDefense,
    AR_TAPE,
)
from src.attacks.fsha import (
    FSHADiscriminator,
    FSHAPilotAutoEncoder,
    FSHAServerAdapter,
    fit_fsha_pilot,
    warmup_fsha_discriminator,
    train_fsha_hijack_epoch,
    evaluate_fsha,
)
from src.metrics import get_lpips_fn, distance_correlation
from src.utils.visualize import save_reconstruction_grid, plot_attack_curves, plot_tradeoff_curves

B0_REF_ACC = 0.9471


class FSHADecoderView(nn.Module):
    """
    Wrapper kết nối ServerAdapter và Pilot Decoder (f_tilde^-1) để tương thích trực tiếp
    với hàm save_reconstruction_grid(client, decoder, loader, ...).
    """
    def __init__(self, pilot_ae, server_adapter=None):
        super().__init__()
        self.pilot_ae = pilot_ae
        self.server_adapter = server_adapter

    def forward(self, z):
        if self.server_adapter is not None:
            z = self.server_adapter(z)
        return self.pilot_ae.decode(z)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bước 4: Tấn công Chủ động FSHA (Feature Space Hijacking Attack) trên B0 - B6"
    )
    parser.add_argument(
        "--milestone",
        choices=["m1", "m2", "m3", "m4", "all"],
        default=None,
        help="Chạy theo từng mũi đo trong kế hoạch: m1 (B0), m2 (B1+B2), m3 (B3), m4 (B4+B5+B6), hoặc all",
    )
    parser.add_argument(
        "--defense",
        choices=["b0", "b1", "b2", "b3", "b4", "b5", "b6", "ar_tape", "all"],
        default="b0",
        help="Lựa chọn cơ chế phòng thủ mục tiêu (mặc định: b0)",
    )
    parser.add_argument("--fit-epochs", type=int, default=30, help="Số epochs Pha 1 Fitting cho Pilot AE (mặc định: 30 theo chuẩn đối xứng với Bước 1/3)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs Pha 2 Hijacking (mặc định: 30 để ghi nhận đầy đủ hiện tượng làm chậm của DP-SGD)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--target-dim", type=int, default=32, help="Chiều không gian mục tiêu d_target của Pilot AE (mặc định: 32)")
    parser.add_argument("--client-lr", type=float, default=0.005, help="Learning rate của Client trong Pha 2 (mặc định: 0.005)")
    parser.add_argument("--disc-lr", type=float, default=1e-3, help="Learning rate của Discriminator D (mặc định: 1e-3)")
    parser.add_argument("--pilot-lr", type=float, default=1e-3, help="Learning rate của Pilot AE ở Pha 1 (mặc định: 1e-3)")
    parser.add_argument("--grad-scale", type=float, default=None, help="Hệ số khuếch đại gradient đối kháng (mặc định: tự động theo mũi đo hoặc 1.0)")
    parser.add_argument("--grad-scales", type=str, default=None, help="Danh sách grad_scale cần quét (vd: 1.0,5.0,15.0,25.0)")
    parser.add_argument("--loss-type", choices=["softplus", "wgan-gp", "bce"], default="softplus", help="Hàm mục tiêu GAN cho Discriminator")
    parser.add_argument("--task-grad-weight", type=float, default=0.0, help="Trọng số trộn gradient tác vụ (0.0 = Pure FSHA chuẩn Pasquini)")
    parser.add_argument("--client-init", choices=["b0", "scratch"], default="b0", help="Khởi tạo Client từ b0 anchor (bị nhiễu nhẹ) hoặc scratch")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá theo epoch (mặc định: 5)")
    parser.add_argument("--priv-ratio", type=float, default=0.5, help="Tỷ lệ dữ liệu riêng tư Client / tổng tập train (mặc định: 0.5)")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2, help="Số luồng DataLoader")
    parser.add_argument("--sweep", action="store_true", default=False, help="Chạy quét toàn bộ dải siêu tham số của từng baseline")
    parser.add_argument("--refit-pilot", action="store_true", default=False, help="Buộc huấn luyện lại Pha 1 Pilot AE kể cả khi đã có checkpoint")
    parser.add_argument("--resume", action="store_true", default=False, help="Tự động bỏ qua các kịch bản đã hoàn thành trong file JSON")

    # Siêu tham số riêng cho từng Baseline
    parser.add_argument("--sigma", type=float, default=None, help="B1: Cường độ nhiễu Gauss đơn lẻ")
    parser.add_argument("--sigmas", type=str, default=None, help="B1: Danh sách sigma (mặc định sweep: 0.1,0.5,1.0)")

    parser.add_argument("--sigma-dp", type=float, default=None, help="B2: Noise multiplier cho DP-SGD")
    parser.add_argument("--sigmas-dp", type=str, default=None, help="B2: Danh sách sigma_dp (mặc định sweep: 0.5,1.0,2.0)")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="B2: Ngưỡng cắt chuẩn gradient C (mặc định: 1.0)")

    parser.add_argument("--alpha", type=float, default=None, help="B3 NoPeek: Hệ số phạt dCor đơn lẻ")
    parser.add_argument("--alphas", type=str, default=None, help="B3 NoPeek: Danh sách alpha (mặc định sweep: 0.1,0.5,1.0,10.0,100.0,500.0)")

    parser.add_argument("--block-size", type=int, default=None, help="B4: Kích thước khối xáo trộn")
    parser.add_argument("--block-sizes", type=str, default=None, help="B4: Danh sách block sizes (mặc định sweep: 2,4,8)")

    parser.add_argument("--distortion-scale", type=float, default=None, help="B5: Hệ số biến dạng lưới tọa độ")
    parser.add_argument("--distortion-scales", type=str, default=None, help="B5: Danh sách distortion scales (mặc định sweep: 0.1,0.2,0.3)")

    parser.add_argument("--b6-alpha", type=float, default=None, help="B6 ADP-AE: Biên độ nhiễu loạn alpha")
    parser.add_argument("--b6-alphas", type=str, default=None, help="B6 ADP-AE: Danh sách alpha (mặc định sweep: 0.05,0.1,0.2)")

    parser.add_argument("--subspace-dim", type=int, default=None, help="AR-TAPE: Số chiều không gian con tác vụ k")
    parser.add_argument("--subspace-dims", type=str, default=None, help="AR-TAPE: Danh sách k (mặc định sweep: 16,32,48)")

    parser.add_argument("--b0-ckpt", type=str, default=None, help="Đường dẫn checkpoint B0 Vanilla")
    parser.add_argument("--b1-dec-ckpt", type=str, default=None, help="Đường dẫn checkpoint Decoder Bước 1")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục dữ liệu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_FSHA"),
        help="Thư mục xuất kết quả FSHA (mặc định: output/AbReTAPE_FSHA)",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Thư mục sao lưu tức thời lên Google Drive sau mỗi kịch bản (vd: /content/drive/MyDrive/AbReTAPE_FSHA)",
    )
    parser.add_argument(
        "--auto-shutdown-colab",
        action="store_true",
        default=False,
        help="Tự động ngắt kết nối và thu hồi máy ảo Google Colab (runtime.unassign) ngay khi chạy xong để tiết kiệm Compute Units",
    )
    return parser.parse_args()


def find_default_ckpt(user_path, candidates):
    if user_path and os.path.isfile(user_path):
        return user_path
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def prepare_client(device, b0_ckpt=None, client_init="b0", init_perturb=0.15, seed=101):
    torch.manual_seed(seed)
    client = ClientModel().to(device)
    if client_init == "b0" and b0_ckpt is not None and os.path.isfile(b0_ckpt):
        ckpt = torch.load(b0_ckpt, map_location=device)
        if "client" in ckpt:
            client.load_state_dict(ckpt["client"])
        if init_perturb > 0.0:
            g = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                for p in client.parameters():
                    if p.ndim == 4:
                        scale = p.std().clamp(min=1e-4)
                        p.add_(init_perturb * scale * torch.randn(p.shape, generator=g, device=device, dtype=p.dtype))
    return client


def save_json_and_csv(out_dir, base_name, records):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    if records:
        csv_path = os.path.join(out_dir, f"{base_name}.csv")
        scalar_keys = [k for k in records[0].keys() if not isinstance(records[0][k], (list, dict))]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_keys)
            writer.writeheader()
            for r in records:
                writer.writerow({k: r.get(k) for k in scalar_keys})


def run_single_fsha_scenario(
    scenario_id,
    defense_code,
    method_name,
    param_label,
    param_key,
    param_val,
    defense_module,
    grad_scale,
    args,
    pilot_ae,
    pilot_server,
    priv_loader,
    pub_loader,
    test_loader,
    lpips_fn,
    b0_ckpt,
    out_dir,
    device,
    use_dpsgd=False,
    sigma_dp=1.0,
    init_perturb=0.02,
):
    print("\n" + "-" * 78)
    print(f">>> [FSHA HIJACKING] KỊCH BẢN: {scenario_id} | {method_name} ({param_label}) | GradScale={grad_scale}x <<<")
    print("-" * 78, flush=True)

    client = prepare_client(
        device=device,
        b0_ckpt=b0_ckpt,
        client_init=args.client_init,
        init_perturb=init_perturb,
    )
    discriminator = FSHADiscriminator(in_channels=64).to(device)
    task_server = ServerModel().to(device)
    if b0_ckpt is not None and os.path.isfile(b0_ckpt):
        ckpt = torch.load(b0_ckpt, map_location=device)
        if "server" in ckpt:
            task_server.load_state_dict(ckpt["server"])

    server_adapter = FSHAServerAdapter(channels=64, defense=defense_module).to(device)

    # Khởi tạo Optimizers
    base_opt_c = torch.optim.SGD(client.parameters(), lr=args.client_lr, momentum=0.9, weight_decay=1e-4)
    if use_dpsgd:
        opt_c = DPSGDClientOptimizer(
            client=client,
            optimizer=base_opt_c,
            max_grad_norm=args.clip_norm,
            noise_multiplier=sigma_dp,
        )
    else:
        opt_c = base_opt_c

    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr, betas=(0.5, 0.999))
    opt_task_s = torch.optim.SGD(task_server.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    opt_adapter = torch.optim.Adam(server_adapter.parameters(), lr=1e-3, weight_decay=1e-5)
    sched_c = CosineAnnealingLR(base_opt_c, T_max=max(args.epochs, 1))

    # Đo trạng thái ban đầu (Trước khi bị FSHA thao túng gradient)
    init_metrics = evaluate_fsha(
        client=client,
        pilot_ae=pilot_ae,
        task_server=task_server,
        test_loader=test_loader,
        device=device,
        defense=defense_module,
        server_adapter=None,
        lpips_fn=lpips_fn,
    )
    print(
        f"  [Before FSHA] PSNR: {init_metrics['psnr']:.2f} dB | SSIM: {init_metrics['ssim']:.4f} | "
        f"MSE: {init_metrics['mse']:.4f} | dCor: {init_metrics['dcor']:.4f} | Task Acc: {init_metrics['test_acc']*100:.2f}%",
        flush=True,
    )

    # Khởi động Discriminator trước khi bơm gradient độc hại
    warmup_fsha_discriminator(
        client=client,
        discriminator=discriminator,
        pilot_ae=pilot_ae,
        priv_loader=priv_loader,
        pub_loader=pub_loader,
        opt_d=opt_d,
        device=device,
        defense=defense_module,
        server_adapter=server_adapter,
        num_batches=15,
        loss_type=args.loss_type,
    )

    history = []
    best_psnr = -1.0
    best_metrics = None
    collapse_epoch_20db = None
    collapse_epoch_25db = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_fsha_hijack_epoch(
            client=client,
            discriminator=discriminator,
            pilot_ae=pilot_ae,
            pilot_server=pilot_server,
            task_server=task_server,
            priv_loader=priv_loader,
            pub_loader=pub_loader,
            opt_c=opt_c,
            opt_d=opt_d,
            opt_task_s=opt_task_s,
            device=device,
            defense=defense_module,
            server_adapter=server_adapter,
            opt_adapter=opt_adapter,
            client_opt_is_dpsgd=use_dpsgd,
            grad_scale=grad_scale,
            loss_type=args.loss_type,
            task_grad_weight=args.task_grad_weight,
        )
        sched_c.step()
        ep_time = time.time() - t0

        should_eval = (epoch % args.eval_freq == 0) or (epoch == 1) or (epoch == args.epochs)
        if should_eval:
            eval_m = evaluate_fsha(
                client=client,
                pilot_ae=pilot_ae,
                task_server=task_server,
                test_loader=test_loader,
                device=device,
                defense=defense_module,
                server_adapter=server_adapter,
                lpips_fn=lpips_fn,
            )
            if collapse_epoch_20db is None and eval_m["psnr"] >= 20.0:
                collapse_epoch_20db = epoch
            if collapse_epoch_25db is None and eval_m["psnr"] >= 25.0:
                collapse_epoch_25db = epoch

            if eval_m["psnr"] > best_psnr:
                best_psnr = eval_m["psnr"]
                best_metrics = dict(eval_m)
                best_metrics["best_epoch"] = epoch

            lpips_str = f"{eval_m['lpips']:.4f}" if eval_m["lpips"] is not None else "N/A"
            print(
                f"  [FSHA Ep {epoch:2d}/{args.epochs}] ({ep_time:.1f}s) | LossD: {train_stats['loss_d']:.4f} | "
                f"SubspaceMSE: {train_stats['loss_subspace']:.4f} | Test Acc: {eval_m['test_acc']*100:.2f}% | "
                f"PSNR: {eval_m['psnr']:.2f} dB | SSIM: {eval_m['ssim']:.4f} | LPIPS: {lpips_str} | dCor: {eval_m['dcor']:.4f}",
                flush=True,
            )
            history.append({
                "epoch": epoch,
                "train_mse": train_stats["loss_subspace"],
                "loss_d": train_stats["loss_d"],
                "loss_adv": train_stats["loss_adv"],
                "test_mse": eval_m["mse"],
                "test_psnr": eval_m["psnr"],
                "test_ssim": eval_m["ssim"],
                "test_lpips": eval_m["lpips"],
                "test_acc": eval_m["test_acc"],
                "dcor": eval_m["dcor"],
                "epoch_time": ep_time,
            })
        else:
            print(
                f"  [FSHA Ep {epoch:2d}/{args.epochs}] ({ep_time:.1f}s) | LossD: {train_stats['loss_d']:.4f} | "
                f"SubspaceMSE: {train_stats['loss_subspace']:.4f} | Train Task Acc: {train_stats['train_task_acc']*100:.2f}%",
                flush=True,
            )
            history.append({
                "epoch": epoch,
                "train_mse": train_stats["loss_subspace"],
                "loss_d": train_stats["loss_d"],
                "loss_adv": train_stats["loss_adv"],
                "epoch_time": ep_time,
            })

    final_metrics = best_metrics if best_metrics is not None else init_metrics

    # Lưu lưới ảnh trực quan hóa và đồ thị hội tụ tấn công
    recon_View = FSHADecoderView(pilot_ae, server_adapter=server_adapter).to(device)
    grid_path = os.path.join(out_dir, f"fsha_recon_{scenario_id}.png")
    curves_path = os.path.join(out_dir, f"fsha_curves_{scenario_id}.png")

    try:
        save_reconstruction_grid(
            client=client,
            decoder=recon_View,
            loader=test_loader,
            device=device,
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
            save_path=grid_path,
            num_images=8,
            defense=defense_module,
        )
        plot_attack_curves(history, save_path=curves_path)
    except Exception as e:
        print(f"  [WARN] Không thể xuất biểu đồ trực quan cho {scenario_id}: {e}")

    is_broken = final_metrics["psnr"] >= 18.0 or final_metrics["ssim"] >= 0.60
    if is_broken:
        verdict = "Sụp đổ trước FSHA (Tái tạo thành công)"
    elif final_metrics["test_acc"] < 0.70:
        verdict = "Chặn được tái tạo nhưng Utility sụp đổ nặng"
    else:
        verdict = "Kháng được FSHA & Giữ vững Utility"

    record = {
        "scenario_id": scenario_id,
        "defense": defense_code.upper(),
        "method": method_name,
        "param_label": param_label,
        param_key: param_val,
        "grad_scale": grad_scale,
        "test_acc": round(final_metrics["test_acc"], 4),
        "delta_acc_vs_b0": round(final_metrics["test_acc"] - B0_REF_ACC, 4),
        "mse": round(final_metrics["mse"], 5),
        "psnr": round(final_metrics["psnr"], 2),
        "ssim": round(final_metrics["ssim"], 4),
        "lpips": round(final_metrics["lpips"], 4) if final_metrics["lpips"] is not None else None,
        "dcor_before": round(init_metrics["dcor"], 4),
        "dcor_after": round(final_metrics["dcor"], 4),
        "psnr_before": round(init_metrics["psnr"], 2),
        "collapse_epoch_20db": collapse_epoch_20db,
        "collapse_epoch_25db": collapse_epoch_25db,
        "best_epoch": final_metrics.get("best_epoch", args.epochs),
        "verdict": verdict,
        "recon_image": os.path.basename(grid_path),
        "history": history,
    }
    if use_dpsgd:
        priv_size = len(priv_loader.dataset)
        eps = compute_dp_epsilon(
            epochs=args.epochs,
            batch_size=args.batch_size,
            dataset_size=priv_size,
            noise_multiplier=sigma_dp,
            delta=1e-5,
        )
        record["dp_epsilon"] = round(eps, 2)

    return record


def build_scenarios_for_defense(defense_code, args, device):
    """
    Xây dựng danh sách cấu hình thực nghiệm cho từng baseline B0-B6 (và AR-TAPE).
    """
    scenarios = []
    d = defense_code.lower()

    if d == "b0":
        if args.grad_scales:
            scales = [float(x.strip()) for x in args.grad_scales.split(",")]
        elif args.grad_scale is not None:
            scales = [args.grad_scale]
        else:
            scales = [1.0, 5.0] if args.sweep else [1.0]

        for gs in scales:
            scenarios.append({
                "scenario_id": f"b0_vanilla_gs{gs}",
                "defense_code": "b0",
                "method_name": "Vanilla Split Learning",
                "param_label": f"Cut-Layer 1 (grad_scale={gs})",
                "param_key": "grad_scale",
                "param_val": gs,
                "defense_module": None,
                "grad_scale": gs,
                "use_dpsgd": False,
                "init_perturb": 0.18,
            })

    elif d == "b1":
        if args.sigmas:
            sigmas = [float(x.strip()) for x in args.sigmas.split(",")]
        elif args.sigma is not None:
            sigmas = [args.sigma]
        else:
            sigmas = [0.1, 0.5, 1.0] if args.sweep else [0.5]

        gs = args.grad_scale if args.grad_scale is not None else 15.0
        for s in sigmas:
            scenarios.append({
                "scenario_id": f"b1_gauss_sigma{s}_gs{gs}",
                "defense_code": "b1",
                "method_name": "Gaussian Noise Injection",
                "param_label": f"sigma={s}, grad_scale={gs}",
                "param_key": "sigma",
                "param_val": s,
                "defense_module": GaussianNoise(sigma=s).to(device),
                "grad_scale": gs,
                "use_dpsgd": False,
                "init_perturb": 0.15,
            })

    elif d == "b2":
        if args.sigmas_dp:
            sigmas_dp = [float(x.strip()) for x in args.sigmas_dp.split(",")]
        elif args.sigma_dp is not None:
            sigmas_dp = [args.sigma_dp]
        else:
            sigmas_dp = [0.5, 1.0, 2.0] if args.sweep else [1.0]

        gs = args.grad_scale if args.grad_scale is not None else 5.0
        for s_dp in sigmas_dp:
            scenarios.append({
                "scenario_id": f"b2_dpsgd_sigma{s_dp}_gs{gs}",
                "defense_code": "b2",
                "method_name": "DP-SGD (Client Optimizer)",
                "param_label": f"sigma_dp={s_dp}, C={args.clip_norm}",
                "param_key": "sigma_dp",
                "param_val": s_dp,
                "defense_module": None,
                "grad_scale": gs,
                "use_dpsgd": True,
                "sigma_dp": s_dp,
                "init_perturb": 0.22,
            })

    elif d == "b3":
        if args.alphas:
            alphas = [float(x.strip()) for x in args.alphas.split(",")]
        elif args.alpha is not None:
            alphas = [args.alpha]
        else:
            alphas = [0.1, 0.5, 1.0, 10.0, 100.0, 500.0] if args.sweep else [0.5, 100.0]

        gs = args.grad_scale if args.grad_scale is not None else 5.0
        for a in alphas:
            eff_gs = 1.0 if a >= 100.0 else gs
            scenarios.append({
                "scenario_id": f"b3_nopeek_alpha{a}_gs{eff_gs}",
                "defense_code": "b3",
                "method_name": "NoPeek (dCor Penalty)",
                "param_label": f"alpha={a}, grad_scale={eff_gs}",
                "param_key": "alpha",
                "param_val": a,
                "defense_module": NoPeekDefense(alpha=a).to(device),
                "grad_scale": eff_gs,
                "use_dpsgd": False,
                "init_perturb": 0.18,
            })

    elif d == "b4":
        if args.block_sizes:
            bs_list = [int(x.strip()) for x in args.block_sizes.split(",")]
        elif args.block_size is not None:
            bs_list = [args.block_size]
        else:
            bs_list = [2, 4, 8] if args.sweep else [4]

        gs = args.grad_scale if args.grad_scale is not None else 5.0
        for bs in bs_list:
            scenarios.append({
                "scenario_id": f"b4_scramble_bs{bs}_gs{gs}",
                "defense_code": "b4",
                "method_name": "Block Scrambling",
                "param_label": f"block_size={bs}, grad_scale={gs}",
                "param_key": "block_size",
                "param_val": bs,
                "defense_module": BlockScrambleDefense(block_size=bs).to(device),
                "grad_scale": gs,
                "use_dpsgd": False,
                "init_perturb": 0.15,
            })

    elif d == "b5":
        if args.distortion_scales:
            d_scales = [float(x.strip()) for x in args.distortion_scales.split(",")]
        elif args.distortion_scale is not None:
            d_scales = [args.distortion_scale]
        else:
            d_scales = [0.1, 0.2, 0.3] if args.sweep else [0.2]

        gs = args.grad_scale if args.grad_scale is not None else 5.0
        for ds in d_scales:
            scenarios.append({
                "scenario_id": f"b5_deform_scale{ds}_gs{gs}",
                "defense_code": "b5",
                "method_name": "Deformable Operator",
                "param_label": f"distortion={ds}, grad_scale={gs}",
                "param_key": "distortion_scale",
                "param_val": ds,
                "defense_module": DeformableOperatorDefense(distortion_scale=ds).to(device),
                "grad_scale": gs,
                "use_dpsgd": False,
                "init_perturb": 0.15,
            })

    elif d == "b6":
        if args.b6_alphas:
            b6_alphas = [float(x.strip()) for x in args.b6_alphas.split(",")]
        elif args.b6_alpha is not None:
            b6_alphas = [args.b6_alpha]
        else:
            b6_alphas = [0.05, 0.1, 0.2] if args.sweep else [0.1, 0.2]

        gs = args.grad_scale if args.grad_scale is not None else 5.0
        for b6_a in b6_alphas:
            b6_mod = ADPAutoEncoderDefense(in_channels=64, alpha=b6_a).to(device)
            for cand_name in [f"b6_ae_alpha_{b6_a}.pt", f"phase1_perturb_ae_alpha{b6_a}.pt"]:
                cand_ckpt = os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step3_B6", cand_name)
                if os.path.isfile(cand_ckpt):
                    try:
                        b6_mod.perturb_ae.load_state_dict(torch.load(cand_ckpt, map_location=device))
                        break
                    except Exception:
                        pass
            for p in b6_mod.parameters():
                p.requires_grad = False

            scenarios.append({
                "scenario_id": f"b6_adp_ae_alpha{b6_a}_gs{gs}",
                "defense_code": "b6",
                "method_name": "ADP-AE (Plug-in AutoEncoder)",
                "param_label": f"alpha={b6_a}, grad_scale={gs}",
                "param_key": "alpha",
                "param_val": b6_a,
                "defense_module": b6_mod,
                "grad_scale": gs,
                "use_dpsgd": False,
                "init_perturb": 0.10,
            })

    elif d == "ar_tape":
        if args.subspace_dims:
            s_dims = [int(x.strip()) for x in args.subspace_dims.split(",")]
        elif args.subspace_dim is not None:
            s_dims = [args.subspace_dim]
        else:
            s_dims = [16, 32, 48] if args.sweep else [24]

        gs = args.grad_scale if args.grad_scale is not None else 5.0
        for k in s_dims:
            scenarios.append({
                "scenario_id": f"ar_tape_k{k}_gs{gs}",
                "defense_code": "ar_tape",
                "method_name": "AR-TAPE (Subspace Projection)",
                "param_label": f"subspace_dim={k}, grad_scale={gs}",
                "param_key": "subspace_dim",
                "param_val": k,
                "defense_module": AR_TAPE(in_channels=64, subspace_dim=k).to(device),
                "grad_scale": gs,
                "use_dpsgd": False,
                "init_perturb": 0.15,
            })

    return scenarios


def print_summary_table(records):
    if not records:
        return
    print("\n" + "=" * 112)
    print("BẢNG TỔNG HỢP KẾT QUẢ TẤN CÔNG CHỦ ĐỘNG FSHA (FEATURE SPACE HIJACKING ATTACK)")
    print("=" * 112)
    header = (
        f"{'Baseline':<8} | {'Cấu hình':<26} | {'Test Acc':>8} | {'PSNR (dB)':>9} | "
        f"{'SSIM':>7} | {'LPIPS':>7} | {'dCor':>7} | {'Ep>=20dB':>8} | {'Nhận định'}"
    )
    print(header)
    print("-" * 112)
    for r in records:
        lpips_s = f"{r['lpips']:.4f}" if r.get("lpips") is not None else "  N/A "
        ep_col = str(r["collapse_epoch_20db"]) if r.get("collapse_epoch_20db") is not None else "Chặn"
        print(
            f"{r['defense']:<8} | {r['param_label']:<26} | {r['test_acc']*100:7.2f}% | "
            f"{r['psnr']:9.2f} | {r['ssim']:7.4f} | {lpips_s:>7} | "
            f"{r['dcor_after']:7.4f} | {ep_col:>8} | {r['verdict']}"
        )
    print("=" * 112 + "\n", flush=True)


def sync_directory(src_dir, dst_dir):
    """
    Sao lưu tức thời toàn bộ tệp kết quả (.json, .csv, .png, .pt) sang thư mục đích (Google Drive)
    và ép hệ điều hành flush buffer xuống đĩa (os.sync).
    """
    if not dst_dir or not os.path.isdir(src_dir):
        return
    import shutil
    try:
        os.makedirs(dst_dir, exist_ok=True)
        for item in os.listdir(src_dir):
            s_path = os.path.join(src_dir, item)
            d_path = os.path.join(dst_dir, item)
            if os.path.isfile(s_path):
                shutil.copy2(s_path, d_path)
        if hasattr(os, "sync"):
            os.sync()
        print(f"  [DRIVE SYNC] Đã sao lưu kết quả an toàn sang: {dst_dir}", flush=True)
    except Exception as e:
        print(f"  [WARN] Lỗi khi sao lưu sang {dst_dir}: {e}", flush=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Nếu có --resume và --backup-dir (Google Drive), khôi phục các file có sẵn từ Drive về out_dir trước khi chạy
    if args.resume and args.backup_dir and os.path.isdir(args.backup_dir):
        print(f"[DRIVE RESTORE] Đang kiểm tra và khôi phục dữ liệu từ: {args.backup_dir} -> {out_dir}")
        sync_directory(args.backup_dir, out_dir)

    print("=" * 85)
    print("BƯỚC 4: TRIỂN KHAI TẤN CÔNG CHỦ ĐỘNG FSHA (PASQUINI ET AL., CCS 2021)")
    print(f"Thiết bị chạy: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Không gian mục tiêu d_target: {args.target_dim} | Pha 1 Fitting: {args.fit_epochs} ep | Pha 2 Hijacking: {args.epochs} ep")
    print(f"Thư mục kết quả: {out_dir}")
    if args.backup_dir:
        print(f"Thư mục sao lưu Google Drive: {args.backup_dir}")
    print("=" * 85, flush=True)

    try:
        # 1. Nạp tập dữ liệu chia tách Private (Client) / Public (Server)
        priv_loader, pub_loader, test_loader = get_cifar10_fsha_splits(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            priv_ratio=args.priv_ratio,
            seed=42,
        )
        print(
            f"[Data Split] Client D_private: {len(priv_loader.dataset)} mẫu | "
            f"Server D_public (Auxiliary): {len(pub_loader.dataset)} mẫu | "
            f"Test set: {len(test_loader.dataset)} mẫu",
            flush=True,
        )

        b0_ckpt = find_default_ckpt(
            args.b0_ckpt,
            [
                os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step0", "b0_vanilla.pt"),
                os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step0", "best_b0_vanilla.pt"),
                os.path.join(PROJECT_ROOT, "b0_vanilla.pt"),
            ],
        )
        b1_dec_ckpt = find_default_ckpt(
            args.b1_dec_ckpt,
            [
                os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step1", "best_b1_decoder.pt"),
                os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step1", "b1_decoder.pt"),
                os.path.join(PROJECT_ROOT, "best_b1_decoder.pt"),
            ],
        )

        lpips_fn = get_lpips_fn(device=device)

        # 2. PHA 1 — FITTING: Xây dựng hoặc nạp lại Khóa Giải Mã của Server (Pilot AutoEncoder)
        pilot_ae = FSHAPilotAutoEncoder(
            in_channels=3, latent_channels=64, target_dim=args.target_dim, hidden_dim=128
        ).to(device)
        pilot_server = ServerModel().to(device)
        pilot_ckpt_path = os.path.join(out_dir, f"pilot_ae_fitted_d{args.target_dim}.pt")

        if os.path.isfile(pilot_ckpt_path) and not args.refit_pilot:
            print(f"\n[Phase 1 Fitting] Tìm thấy checkpoint Khóa Giải Mã tại: {pilot_ckpt_path}. Đang nạp...")
            p_ckpt = torch.load(pilot_ckpt_path, map_location=device)
            pilot_ae.load_state_dict(p_ckpt["pilot_ae"])
            if p_ckpt.get("pilot_server") is not None:
                pilot_server.load_state_dict(p_ckpt["pilot_server"])
            pilot_ae.eval()
            pilot_server.eval()
            for p in pilot_ae.parameters():
                p.requires_grad = False
            for p in pilot_server.parameters():
                p.requires_grad = False
            print("[Phase 1 Fitting] Đã nạp xong Pilot AE & Pilot Server!", flush=True)
        else:
            print(f"\n[Phase 1 Fitting] Bắt đầu huấn luyện Pilot AE (d_target={args.target_dim}) trên D_public...")
            fit_fsha_pilot(
                pilot_ae=pilot_ae,
                pilot_server=pilot_server,
                pub_loader=pub_loader,
                test_loader=test_loader,
                epochs=args.fit_epochs,
                lr=args.pilot_lr,
                device=device,
                b0_ckpt=b0_ckpt,
                b1_dec_ckpt=b1_dec_ckpt,
                save_path=pilot_ckpt_path,
            )
            sync_directory(out_dir, args.backup_dir)

        # 3. Xác định danh sách các Baseline cần chạy theo --milestone hoặc --defense
        if args.milestone is not None:
            m = args.milestone.lower()
            if m == "m1":
                target_defenses = ["b0"]
            elif m == "m2":
                target_defenses = ["b1", "b2"]
            elif m == "m3":
                target_defenses = ["b3"]
            elif m == "m4":
                target_defenses = ["b4", "b5", "b6"]
            else:
                target_defenses = ["b0", "b1", "b2", "b3", "b4", "b5", "b6"]
        else:
            if args.defense.lower() == "all":
                target_defenses = ["b0", "b1", "b2", "b3", "b4", "b5", "b6"]
            else:
                target_defenses = [args.defense.lower()]

        master_json_path = os.path.join(out_dir, "fsha_comprehensive_results.json")
        all_records = []
        if args.resume and os.path.isfile(master_json_path):
            try:
                with open(master_json_path, "r", encoding="utf-8") as f:
                    all_records = json.load(f)
            except Exception:
                all_records = []

        completed_ids = {r.get("scenario_id") for r in all_records if "scenario_id" in r}

        # 4. PHA 2 — HIJACKING: Thực thi FSHA trên từng kịch bản
        for def_code in target_defenses:
            scenarios = build_scenarios_for_defense(def_code, args, device)
            def_records = [r for r in all_records if r.get("defense", "").lower() == def_code]

            for sc in scenarios:
                sc_id = sc["scenario_id"]
                if args.resume and sc_id in completed_ids:
                    print(f"\n[RESUME] Kịch bản '{sc_id}' đã hoàn tất trước đó. Bỏ qua.")
                    continue

                rec = run_single_fsha_scenario(
                    scenario_id=sc_id,
                    defense_code=sc["defense_code"],
                    method_name=sc["method_name"],
                    param_label=sc["param_label"],
                    param_key=sc["param_key"],
                    param_val=sc["param_val"],
                    defense_module=sc["defense_module"],
                    grad_scale=sc["grad_scale"],
                    args=args,
                    pilot_ae=pilot_ae,
                    pilot_server=pilot_server,
                    priv_loader=priv_loader,
                    pub_loader=pub_loader,
                    test_loader=test_loader,
                    lpips_fn=lpips_fn,
                    b0_ckpt=b0_ckpt,
                    out_dir=out_dir,
                    device=device,
                    use_dpsgd=sc.get("use_dpsgd", False),
                    sigma_dp=sc.get("sigma_dp", 1.0),
                    init_perturb=sc.get("init_perturb", 0.15),
                )
                def_records.append(rec)
                all_records = [r for r in all_records if r.get("scenario_id") != sc_id] + [rec]

                save_json_and_csv(out_dir, f"results_fsha_{def_code}", def_records)
                save_json_and_csv(out_dir, "fsha_comprehensive_results", all_records)
                sync_directory(out_dir, args.backup_dir)

            if len(def_records) > 1:
                p_key = scenarios[0]["param_key"]
                tradeoff_plot = os.path.join(out_dir, f"fsha_{def_code}_tradeoff.png")
                try:
                    plot_tradeoff_curves(def_records, save_path=tradeoff_plot, x_key=p_key, x_label=p_key)
                except Exception:
                    pass
                sync_directory(out_dir, args.backup_dir)

        print_summary_table(all_records)
    finally:
        sync_directory(out_dir, args.backup_dir)
        if args.auto_shutdown_colab:
            print("\n[AUTO-SHUTDOWN] Đang chờ 5 giây để đồng bộ hoàn tất Google Drive trước khi tắt Colab Runtime...", flush=True)
            time.sleep(5)
            try:
                from google.colab import runtime
                print("[AUTO-SHUTDOWN] Đang gọi runtime.unassign() để giải phóng GPU!", flush=True)
                runtime.unassign()
            except Exception as e:
                print(f"[WARN] Không thể gọi google.colab.runtime.unassign(): {e}", flush=True)


if __name__ == "__main__":
    main()
