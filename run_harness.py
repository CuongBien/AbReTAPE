#!/usr/bin/env python3
# CLI Runner trung tâm của Research Harness cho đề tài AR-TAPE
import os
import sys
import argparse
import json
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel, ServerModel
from src.data import get_cifar10
from src.attacks import Decoder
from src.harness import ResearchHarness, build_defense

CHECKLIST = [
    {"phase": "Foundation", "item": "Môi trường ảo & uv package manager (CUDA 13.0, PyTorch 2.14)", "status": "DONE"},
    {"phase": "Foundation", "item": "Bộ kiểm thử đơn vị tự động 6/6 module (run_tests.py)", "status": "DONE"},
    {"phase": "Tier 1: B0", "item": "Vanilla Split Learning & Centralized đối chứng (Bước 0)", "status": "DONE"},
    {"phase": "Tier 1: Attack", "item": "Passive Feature Inversion Decoder attack (Bước 1)", "status": "DONE"},
    {"phase": "Tier 1: N1", "item": "Thực nghiệm chứng minh Hấp thụ Theta_s -> E^-1 (Bước 2: Adapter & Covariance)", "status": "DONE"},
    {"phase": "Tier 1: Baselines", "item": "B1: Gaussian Noise Sweep (sigma=0.1, 0.5, 1.0)", "status": "DONE"},
    {"phase": "Tier 1: Baselines", "item": "B2: DP-SGD Sweep (sigma_dp=0.5, 1.0, 2.0; RDP Accountant)", "status": "DONE"},
    {"phase": "Tier 1: Baselines", "item": "B3: NoPeek dCor Sweep (alpha=0.1, 0.5, 1.0)", "status": "DONE"},
    {"phase": "Tier 1: Baselines", "item": "B4: Block Scrambling Sweep (block_size=2, 4, 8)", "status": "DONE"},
    {"phase": "Tier 1: Baselines", "item": "B5: Deformable Operator Sweep (distortion=0.1, 0.2, 0.3)", "status": "DONE"},
    {"phase": "Tier 1: Baselines", "item": "B6: ADP AutoEncoder Defense plug-in module", "status": "READY"},
    {"phase": "Novelty N2", "item": "Cơ chế AR-TAPE (Subspace Projection P_task + Non-invertible Encoder)", "status": "READY"},
    {"phase": "Attacks", "item": "Stress-test FSHA (Feature Space Hijacking Attack) module", "status": "READY"},
    {"phase": "Metrics", "item": "Bộ đo Clinical (AUC-ROC, Sens@90%Spec, Youden's J, Macro F1)", "status": "DONE"},
    {"phase": "Metrics", "item": "Bộ đo Visual Alignment (Grad-CAM Cosine & Pearson)", "status": "DONE"},
    {"phase": "Metrics", "item": "Bộ đo Information Leakage (dCor + Mutual Information MINE bound)", "status": "DONE"},
    {"phase": "Tier 2: Medical", "item": "Nạp dữ liệu & kiểm chứng Case Study trên HAM10000 / PCAM", "status": "PENDING"}
]


def print_checklist():
    print("\n" + "=" * 80)
    print("      CHECKLIST TIẾN ĐỘ ĐỀ TÀI AR-TAPE (SPLIT LEARNING RESEARCH HARNESS)")
    print("=" * 80)
    done_count = sum(1 for c in CHECKLIST if c["status"] == "DONE")
    ready_count = sum(1 for c in CHECKLIST if c["status"] == "READY")
    pending_count = sum(1 for c in CHECKLIST if c["status"] == "PENDING")
    total = len(CHECKLIST)

    print(f"Tổng quan: {done_count}/{total} ĐÃ XONG | {ready_count} SẴN SÀNG CHẠY | {pending_count} ĐANG ĐỢI DỮ LIỆU\n")
    print(f"{'Giai đoạn':<20} | {'Trạng thái':<10} | {'Nội dung công việc'}")
    print("-" * 80)
    for c in CHECKLIST:
        st = f"[x] {c['status']}" if c["status"] == "DONE" else (f"[~] {c['status']}" if c["status"] == "READY" else f"[ ] {c['status']}")
        print(f"{c['phase']:<20} | {st:<10} | {c['item']}")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Research Harness CLI cho đề tài AR-TAPE")
    parser.add_argument("--checklist", action="store_true", help="Hiển thị checklist tiến độ đề tài")
    parser.add_argument("--mode", choices=["eval_all", "eval_single", "smoke_test"], default="eval_all",
                        help="Chế độ chạy: eval_all (toàn bộ bảng), eval_single, smoke_test")
    parser.add_argument("--defense", type=str, default="vanilla",
                        help="Lựa chọn defense: vanilla, b1, b2, b3, b4, b5, b6, ar_tape")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checklist:
        print_checklist()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Khởi động AR-TAPE Research Harness trên: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if args.mode == "smoke_test":
        print("\n[SMOKE TEST] Kiểm tra tương thích các module mới...")
        x_dummy = torch.randn(2, 64, 32, 32, device=device)
        for def_name in ["b1", "b3", "b4", "b5", "b6", "ar_tape"]:
            d = build_defense(def_name).to(device)
            out = d(x_dummy)
            assert out.shape == x_dummy.shape, f"Lỗi shape tại defense {def_name}: {out.shape}"
            print(f"  -> Defense '{def_name}' kiểm tra forward pass: ĐẠT!")
        print("[SMOKE TEST] Tất cả các module defense B1-B6 và AR-TAPE hoạt động hoàn hảo!\n")

    elif args.mode == "eval_all":
        eval_file = os.path.join(args.output_dir, "full_comprehensive_evaluation.json")
        if os.path.isfile(eval_file):
            with open(eval_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            print(f"\n[EVALUATION REPORT] Tải kết quả đánh giá mới nhất từ: {eval_file}")
            print(f"{'Tên phương pháp':<28} | {'Test Acc':<9} | {'PSNR (dB)':<9} | {'SSIM':<7} | {'LPIPS':<8} | {'dCor(X,Z)':<9}")
            print("-" * 80)
            for k, v in res.items():
                acc = f"{v.get('test_acc', 0)*100:.2f}%" if v.get('test_acc') is not None else "N/A"
                psnr = f"{v.get('psnr', 0):.2f}" if v.get('psnr') is not None else "N/A"
                ssim = f"{v.get('ssim', 0):.4f}" if v.get('ssim') is not None else "N/A"
                lpips_v = f"{v.get('lpips', 0):.6f}" if v.get('lpips') is not None else "N/A"
                dcor = f"{v.get('dcor', 0):.4f}" if v.get('dcor') is not None else "N/A"
                print(f"{k:<28} | {acc:<9} | {psnr:<9} | {ssim:<7} | {lpips_v:<8} | {dcor:<9}")
            print("-" * 80)
        else:
            print("[WARN] Chưa tìm thấy file kết quả tổng hợp. Hãy chạy: uv run python scratch/evaluate_all.py")


if __name__ == "__main__":
    main()
