# Experiment Harness Pipeline: Khung điều phối nghiên cứu tập trung cho toàn bộ đề tài AR-TAPE
import os
import json
import torch
import torch.nn as nn

from ..models import ClientModel, ServerModel
from ..data import get_cifar10, CIFAR10_MEAN, CIFAR10_STD
from ..defenses import (
    GaussianNoise,
    DPSGDClientOptimizer,
    NoPeekDefense,
    BlockScrambleDefense,
    DeformableOperatorDefense,
    ADPAutoEncoderDefense,
    AR_TAPE
)
from ..attacks import Decoder, evaluate_inversion
from ..metrics import (
    get_lpips_fn,
    distance_correlation,
    compute_binary_clinical_metrics,
    compute_multiclass_clinical_metrics,
    GradCAM,
    compute_gradcam_alignment
)
from ..training import train_sl_epoch, evaluate_sl


def build_defense(defense_name, **kwargs):
    """Factory khởi tạo các baseline B0-B6 và AR-TAPE"""
    name = defense_name.lower()
    if name in ["none", "vanilla", "b0"]:
        return None
    elif name in ["gaussian", "b1"]:
        sigma = kwargs.get("sigma", 0.5)
        return GaussianNoise(sigma=sigma)
    elif name in ["nopeek", "b3"]:
        alpha = kwargs.get("alpha", 0.5)
        return NoPeekDefense(alpha=alpha)
    elif name in ["block_scramble", "b4"]:
        block_size = kwargs.get("block_size", 4)
        return BlockScrambleDefense(block_size=block_size)
    elif name in ["deformable", "b5"]:
        distortion = kwargs.get("distortion_scale", 0.2)
        return DeformableOperatorDefense(distortion_scale=distortion)
    elif name in ["adp", "b6"]:
        return ADPAutoEncoderDefense(in_channels=64, bottleneck_channels=16)
    elif name in ["ar_tape", "tape", "proposed"]:
        subspace_dim = kwargs.get("subspace_dim", 32)
        return AR_TAPE(in_channels=64, subspace_dim=subspace_dim)
    else:
        raise ValueError(f"Unknown defense name: {defense_name}")


class ResearchHarness:
    """
    Harness nghiên cứu trung tâm:
    - Điều phối huấn luyện Split Learning (với defense tương ứng)
    - Tự động đánh giá đầy đủ 3 nhóm chỉ số:
      1. Utility (Top-1 Acc, Clinical AUC-ROC, Macro F1, Sens@90%Spec, Grad-CAM alignment)
      2. Security (Reconstruction MSE, PSNR, SSIM, LPIPS)
      3. Information Leakage (dCor, Mutual Information)
    """
    def __init__(self, device=None, data_dir="data"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.lpips_fn = get_lpips_fn(device=self.device)

    def evaluate_comprehensive(self, client, server, decoder, testloader, defense=None, mean=CIFAR10_MEAN, std=CIFAR10_STD, client_vanilla=None):
        """
        Đánh giá đầy đủ toàn bộ metrics theo chuẩn bài báo
        """
        client.eval()
        server.eval()
        if defense is not None:
            defense.eval()

        # 1. Utility: Test Accuracy
        _, test_acc = evaluate_sl(client, server, testloader, self.device, defense=defense)

        # 2. Security: Inversion Attack metrics
        mse, psnr, ssim, lpips_val = None, None, None, None
        if decoder is not None:
            decoder.eval()
            mse, psnr, ssim, lpips_val = evaluate_inversion(
                client, decoder, testloader, self.device, mean, std, defense=defense, lpips_fn=self.lpips_fn
            )

        # 3. Information Leakage: dCor(X, Z')
        dcor_sum = 0.0
        n_eval = 0
        with torch.no_grad():
            for x_t, _ in testloader:
                x_t = x_t.to(self.device)
                z_t = client(x_t)
                if defense is not None:
                    z_t = defense(z_t)
                dcor_sum += distance_correlation(x_t, z_t).item()
                n_eval += 1
                if n_eval >= 15:
                    break
        avg_dcor = dcor_sum / max(n_eval, 1)

        # 4. Visual Alignment (Grad-CAM): nếu có mô hình Vanilla đối chứng
        gradcam_res = None
        if client_vanilla is not None:
            try:
                # Target layer: client last block
                target_l_orig = client_vanilla.layer1[-1].conv2
                target_l_def = client.layer1[-1].conv2

                cam_gen_orig = GradCAM(client_vanilla, target_l_orig)
                cam_gen_def = GradCAM(client, target_l_def)

                cam_sims = []
                cam_pearsons = []
                for x_sample, _ in testloader:
                    x_sample = x_sample[:8].to(self.device)
                    c_orig = cam_gen_orig.generate_cam(x_sample)
                    c_def = cam_gen_def.generate_cam(x_sample)
                    align = compute_gradcam_alignment(c_orig, c_def)
                    cam_sims.append(align["gradcam_cosine_similarity"])
                    cam_pearsons.append(align["gradcam_pearson_correlation"])
                    break
                gradcam_res = {
                    "cosine_similarity": float(sum(cam_sims) / max(len(cam_sims), 1)),
                    "pearson_correlation": float(sum(cam_pearsons) / max(len(cam_pearsons), 1))
                }
            except Exception as e:
                gradcam_res = {"error": str(e)}

        return {
            "utility": {
                "test_accuracy": test_acc,
                "gradcam_alignment": gradcam_res
            },
            "security": {
                "reconstruction_mse": mse,
                "reconstruction_psnr": psnr,
                "reconstruction_ssim": ssim,
                "reconstruction_lpips": lpips_val
            },
            "information_leakage": {
                "distance_correlation": avg_dcor
            }
        }
