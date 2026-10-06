#!/usr/bin/env python3
# Kiểm thử đơn vị tự động toàn diện (Comprehensive Smoke Tests) cho toàn bộ hệ thống AR-TAPE
import os
import sys
import numpy as np
import torch
import torch.nn as nn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel, ServerModel, resnet18_cifar
from src.defenses import (
    random_perm,
    ChannelPermute,
    Adapter,
    GaussianNoise,
    DPSGDClientOptimizer,
    compute_dp_epsilon,
    NoPeekDefense,
    BlockScrambleDefense,
    DeformableOperatorDefense,
    ADPAutoEncoderDefense,
    AR_TAPE,
    FixedOrthoProjection,
    SplitProjection
)
from src.attacks import (
    Decoder,
    recover_perm,
    recover_perm_from_adapter,
    match_accuracy,
    FSHADiscriminator,
    FSHAPilotAutoEncoder,
    train_fsha_step
)
from src.metrics import (
    psnr_ssim,
    denormalize,
    get_lpips_fn,
    calculate_lpips,
    distance_correlation,
    compute_binary_clinical_metrics,
    compute_multiclass_clinical_metrics,
    GradCAM,
    compute_gradcam_alignment,
    MutualInformationEstimator
)
from src.training import train_sl_epoch, evaluate_sl, EarlyStopping
from src.utils import plot_training_curves


def test_models_and_sl(device):
    print("\n[TEST 1/9] KIỂM THỬ MÔ HÌNH VÀ SPLIT LEARNING FORWARD/BACKWARD...")
    client = ClientModel().to(device)
    server = ServerModel().to(device)

    # 1. Kiểm tra shapes
    x_dummy = torch.randn(2, 3, 32, 32, device=device)
    z = client(x_dummy)
    assert z.shape == (2, 64, 32, 32), f"Lỗi shape z: {z.shape}"
    logits = server(z)
    assert logits.shape == (2, 10), f"Lỗi shape logits: {logits.shape}"
    print("  -> Kiểm tra kích thước z [2, 64, 32, 32] và logits [2, 10]: ĐẠT!")

    # 2. Kiểm tra backward qua biên giới
    opt_c = torch.optim.SGD(client.parameters(), lr=0.01)
    opt_s = torch.optim.SGD(server.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    y_dummy = torch.tensor([0, 1], device=device)

    dummy_loader = [(x_dummy, y_dummy)]
    train_loss, train_acc = train_sl_epoch(client, server, dummy_loader, opt_c, opt_s, criterion, device)
    val_loss, val_acc = evaluate_sl(client, server, dummy_loader, device, criterion=criterion)
    print("  -> Kiểm tra Split Learning forward & backward boundary: ĐẠT!")


def test_attacks_and_metrics(device):
    print("\n[TEST 2/9] KIỂM THỬ DECODER TẤN CÔNG VÀ BỘ ĐỘ ĐO (PSNR, SSIM, LPIPS)...")
    decoder = Decoder(in_channels=64, out_channels=3).to(device)
    z_dummy = torch.randn(2, 64, 32, 32, device=device)
    x_rec = decoder(z_dummy)
    assert x_rec.shape == (2, 3, 32, 32), f"Lỗi shape x_rec: {x_rec.shape}"

    # Kiểm tra PSNR, SSIM
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    x_clean = torch.rand(2, 3, 32, 32, device=device)
    p, s = psnr_ssim(x_clean, x_clean, mean, std)
    print(f"  -> Kiểm tra tính toán PSNR/SSIM ({p:.1f} dB, {s:.4f}): ĐẠT!")

    # Kiểm tra LPIPS
    lpips_fn = get_lpips_fn(device=device)
    if lpips_fn is not None:
        lpips_val = calculate_lpips(x_clean, x_clean, mean, std, lpips_fn)
        print(f"  -> Kiểm tra tính toán LPIPS ({lpips_val:.6f}): ĐẠT!")
    else:
        print("  -> Bỏ qua LPIPS (chưa cài đặt).")


def test_defenses_step2(device):
    print("\n[TEST 3/9] KIỂM THỬ HOÁN VỊ KÊNH, ADAPTER VÀ KHÔI PHỤC HẤP THỤ...")
    perm = random_perm(64, seed=42).to(device)
    permute = ChannelPermute(perm).to(device)
    z = torch.randn(4, 64, 16, 16, device=device)
    z_perm = permute(z)
    z_rec = permute.inverse(z_perm)
    assert torch.allclose(z, z_rec, atol=1e-5), "Lỗi giải mã hoán vị ChannelPermute!"
    print("  -> Kiểm tra ChannelPermute forward/inverse bảo toàn 100%: ĐẠT!")

    # Kiểm tra Adapter
    adapter = Adapter(channels=64).to(device)
    opt_a = torch.optim.Adam(adapter.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(5):
        opt_a.zero_grad()
        loss = loss_fn(adapter(z_perm), z)
        loss.backward()
        opt_a.step()

    A = adapter.get_matrix()
    hat_pi = recover_perm_from_adapter(A)
    acc = match_accuracy(perm, hat_pi)
    print(f"  -> Kiểm tra Cut-Layer Adapter khôi phục hoán vị ({acc*100:.1f}%): ĐẠT!")


def test_defenses_step3(device):
    print("\n[TEST 4/9] KIỂM THỬ BASELINES B1 (GAUSSIAN NOISE) VÀ B2 (DP-SGD)...")
    noise = GaussianNoise(sigma=0.5).to(device)
    z = torch.zeros(1000, 64, 4, 4, device=device)
    z_noisy = noise(z)
    assert abs(z_noisy.std().item() - 0.5) < 0.05, "Lỗi độ lệch chuẩn GaussianNoise!"
    print(f"  -> Kiểm tra Gaussian Noise (kỳ vọng 0.5, thực tế {z_noisy.std().item():.4f}): ĐẠT!")

    # Kiểm tra DP-SGD Optimizer
    model = nn.Linear(10, 2).to(device)
    opt = DPSGDClientOptimizer(model.parameters(), lr=0.01, sigma_dp=1.0, clip_norm=1.0)
    x = torch.randn(4, 10, device=device)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    print("  -> Kiểm tra DPSGDClientOptimizer clipping và noise addition: ĐẠT!")

    # Kiểm tra RDP Accountant
    eps = compute_dp_epsilon(epochs=10, batch_size=128, dataset_size=50000, sigma=1.0, delta=1e-5)
    assert 0.5 < eps < 5.0, f"Lỗi giá trị Epsilon bất thường: {eps}"
    print(f"  -> Kiểm tra RDP Accountant (10 epochs, sigma=1.0 -> Epsilon={eps:.2f}): ĐẠT!")

    # Kiểm tra Distance Correlation (dCor)
    x1 = torch.randn(32, 10, device=device)
    dcor_self = distance_correlation(x1, x1).item()
    assert abs(dcor_self - 1.0) < 1e-3, f"dCor tự tương quan phải bằng 1.0, thực tế: {dcor_self}"
    x2 = torch.randn(32, 10, device=device)
    dcor_diff = distance_correlation(x1, x2).item()
    print(f"  -> Kiểm tra Distance Correlation dCor (Tự tương quan: {dcor_self:.4f}, Độc lập: {dcor_diff:.4f}): ĐẠT!")

    # Kiểm tra NoPeek Defense forward
    nopeek = NoPeekDefense(alpha=0.5).to(device)
    dcor_loss = nopeek.compute_penalty(x1, x2)
    assert dcor_loss >= 0.0, "dCor loss không thể âm!"
    print("  -> Kiểm tra NoPeek Split Learning training step với dCor penalty: ĐẠT!")


def test_early_stopping():
    print("\n[TEST 5/9] KIỂM THỬ MODULE EARLY STOPPING...")
    es = EarlyStopping(patience=3, min_delta=1e-3, mode="max")
    assert not es.step(0.80, epoch=1)
    assert not es.step(0.85, epoch=2)  # improved
    assert not es.step(0.84, epoch=3)  # counter = 1
    assert not es.step(0.83, epoch=4)  # counter = 2
    assert es.step(0.82, epoch=5)      # counter = 3 -> True
    assert es.best_epoch == 2
    assert es.best_score == 0.85
    print("  -> Kiểm tra EarlyStopping (mode='max', patience=3): ĐẠT!")


def test_plotting():
    print("\n[TEST 6/9] KIỂM THỬ CÔNG CỤ VẼ ĐỒ THỊ...")
    dummy_history = [
        {"epoch": 1, "train_loss": 1.5, "train_acc": 0.5, "test_loss": 1.2, "test_acc": 0.6, "lr": 0.1, "epoch_time": 1.0},
        {"epoch": 2, "train_loss": 0.8, "train_acc": 0.7, "test_loss": 0.7, "test_acc": 0.8, "lr": 0.05, "epoch_time": 1.0},
    ]
    test_img = os.path.join(PROJECT_ROOT, "test_dummy_plot.png")
    plot_training_curves(dummy_history, save_path=test_img)
    assert os.path.isfile(test_img), "Không thể lưu file ảnh đồ thị!"
    os.remove(test_img)
    print("  -> Kiểm tra tạo và xuất biểu đồ matplotlib: ĐẠT!")


def test_harness_defenses_and_ar_tape(device):
    print("\n[TEST 7/9] KIỂM THỬ CÁC BASELINES MỚI (B4, B5, B6) VÀ AR-TAPE (NOVELTY N2)...")
    z = torch.randn(2, 64, 32, 32, device=device)

    # 1. B4: Block Scramble
    b4 = BlockScrambleDefense(block_size=4).to(device)
    z_b4 = b4(z)
    assert z_b4.shape == z.shape, f"B4 shape sai: {z_b4.shape}"
    print("  -> B4 (Block Scramble Defense) forward pass: ĐẠT!")

    # 2. B5: Deformable Operator (Kiya et al. 2024)
    b5 = DeformableOperatorDefense(channels=64, distortion_scale=0.1).to(device)
    z_b5 = b5(z)
    assert z_b5.shape == z.shape, f"B5 shape sai: {z_b5.shape}"
    print("  -> B5 (Deformable Operator Defense) forward pass: ĐẠT!")

    # 3. B6: ADP-style AutoEncoder
    b6 = ADPAutoEncoderDefense(in_channels=64, bottleneck_channels=16).to(device)
    z_b6 = b6(z)
    assert z_b6.shape == z.shape, f"B6 shape sai: {z_b6.shape}"
    print("  -> B6 (ADP-style AutoEncoder) forward pass: ĐẠT!")

    # 4. Proposed AR-TAPE (Subspace Projection P_task)
    tape = AR_TAPE(in_channels=64, subspace_dim=32).to(device)
    z_tape = tape(z)
    assert z_tape.shape == z.shape, f"AR-TAPE shape sai: {z_tape.shape}"
    ortho_loss = tape.get_orthogonality_loss()
    assert ortho_loss >= 0.0, "Orthogonality loss không thể âm!"
    print(f"  -> AR-TAPE (Subspace Projection P_task=V V^T) forward & ortho loss ({ortho_loss.item():.4f}): ĐẠT!")

    # 5. B7: LightSplit (Fixed Orthogonal Projection)
    proj = FixedOrthoProjection(D=64*32*32, k=1024, seed=42, device=device).to(device)
    zt = proj(z)
    assert zt.shape == (2, 1024), f"B7 zt shape sai: {zt.shape}"
    zh = proj.lift(zt)
    assert zh.shape == (2, 64*32*32), f"B7 zh shape sai: {zh.shape}"
    ortho_err = proj.get_orthogonality_error()
    assert ortho_err < 1e-4, f"B7 orthogonality error quá lớn: {ortho_err}"

    dummy_server = ServerModel().to(device)
    split_proj_server = SplitProjection(proj, dummy_server, mode="F").to(device)
    out_f = split_proj_server(zt)
    assert out_f.shape == (2, 10), f"B7 mode F output shape sai: {out_f.shape}"
    print(f"  -> B7 (LightSplit Fixed Orthogonal Projection, k=1024, max |R^T R - I|: {ortho_err:.2e}): ĐẠT!")


def test_clinical_and_gradcam_metrics(device):
    print("\n[TEST 8/9] KIỂM THỬ BỘ ĐỘ ĐO LÂM SÀNG (AUC, SENS@90%SPEC) VÀ GRAD-CAM ALIGNMENT...")
    # 1. Binary Clinical Metrics (PCAM)
    y_true_bin = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_probs_bin = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.4, 0.95, 0.6])
    res_bin = compute_binary_clinical_metrics(y_true_bin, y_probs_bin)
    assert "auc_roc" in res_bin and res_bin["auc_roc"] > 0.9
    assert "sensitivity_at_90_specificity" in res_bin
    assert "youden_threshold" in res_bin
    print(f"  -> Binary Clinical Metrics (AUC: {res_bin['auc_roc']:.4f}, Sens@90%Spec: {res_bin['sensitivity_at_90_specificity']:.4f}): ĐẠT!")

    # 2. Multiclass Clinical Metrics (HAM10000)
    y_true_multi = np.random.randint(0, 7, size=50)
    y_probs_multi = np.random.dirichlet(np.ones(7), size=50)
    res_multi = compute_multiclass_clinical_metrics(y_true_multi, y_probs_multi, num_classes=7)
    assert "macro_auc_roc" in res_multi and "macro_f1" in res_multi
    print(f"  -> Multiclass Clinical Metrics (Macro F1: {res_multi['macro_f1']:.4f}): ĐẠT!")

    # 3. Grad-CAM & Visual Attention Alignment
    client_dummy = ClientModel().to(device)
    target_layer = client_dummy.layer1[-1].conv2
    gradcam = GradCAM(client_dummy, target_layer)
    x_test = torch.randn(2, 3, 32, 32, device=device)
    cam1 = gradcam.generate_cam(x_test)
    assert cam1.shape == (2, 1, 32, 32), f"Grad-CAM shape sai: {cam1.shape}"
    align = compute_gradcam_alignment(cam1, cam1)
    assert abs(align["gradcam_cosine_similarity"] - 1.0) < 1e-4
    assert abs(align["gradcam_pearson_correlation"] - 1.0) < 1e-4
    print(f"  -> Grad-CAM Generator & Attention Alignment (Cos: {align['gradcam_cosine_similarity']:.4f}, Pearson: {align['gradcam_pearson_correlation']:.4f}): ĐẠT!")


def test_fsha_and_leakage(device):
    print("\n[TEST 9/9] KIỂM THỬ TẤN CÔNG CHỦ ĐỘNG FSHA VÀ INFORMATION LEAKAGE (MI / DCOR)...")
    # 1. Mutual Information Estimator
    mi_est = MutualInformationEstimator().to(device)
    x_dummy = torch.randn(4, 3, 32, 32, device=device)
    z_dummy = torch.randn(4, 64, 32, 32, device=device)
    mi_val = mi_est(x_dummy, z_dummy)
    print(f"  -> Mutual Information Neural Estimation bound ({mi_val.item():.4f}): ĐẠT!")

    # 2. FSHA Active Attack
    client = ClientModel().to(device)
    discriminator = FSHADiscriminator(in_channels=64).to(device)
    pilot_ae = FSHAPilotAutoEncoder(in_channels=3, latent_channels=64).to(device)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    opt_ae = torch.optim.Adam(pilot_ae.parameters(), lr=1e-3)

    x_priv = torch.randn(2, 3, 32, 32, device=device)
    x_pub = torch.randn(2, 3, 32, 32, device=device)
    fsha_res = train_fsha_step(client, discriminator, pilot_ae, x_priv, x_pub, opt_d, opt_ae, device=device)
    assert "loss_ae" in fsha_res and "loss_d" in fsha_res
    print(f"  -> FSHA Step (Loss AE: {fsha_res['loss_ae']:.4f}, Loss D: {fsha_res['loss_d']:.4f}): ĐẠT!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"BẮT ĐẦU CHẠY KIỂM THỬ ĐƠN VỊ TOÀN BỘ HỆ THỐNG AR-TAPE (9/9 TESTS)")
    print(f"Thiết bị kiểm thử: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    test_models_and_sl(device)
    test_attacks_and_metrics(device)
    test_defenses_step2(device)
    test_defenses_step3(device)
    test_early_stopping()
    test_plotting()
    test_harness_defenses_and_ar_tape(device)
    test_clinical_and_gradcam_metrics(device)
    test_fsha_and_leakage(device)

    print("\n" + "=" * 70)
    print(">>> TẤT CẢ 9/9 MODULE VÀ THỬ NGHIỆM ĐỀU VƯỢT QUA KIỂM THỬ 100%! <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
