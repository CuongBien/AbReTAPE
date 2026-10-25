# CIFAR-10 PoC Research Archive & Step-by-Step Replication

Thư mục này lưu trữ toàn bộ các kịch bản thực nghiệm, ghi chép cấu hình và bằng chứng khoa học của **Giai đoạn 1 (PoC Verification trên CIFAR-10)** cho đề tài **AR-TAPE**.

---

## 1. Mục Tiêu Đã Hoàn Thành Của Giai Đoạn PoC
1. **Kiểm chứng hiện tượng Hấp thụ Ánh xạ Ngược (Absorption Recovery $\theta_s \to E^{-1}$):**
   - Đã chứng minh trên các phương pháp mã hóa affine (B1, B4, B5), Server độc hại có thể tự động học lại ánh xạ ngược thông qua gradient cập nhật động.
2. **Đối chuẩn toàn diện các cơ chế phòng thủ:**
   - **B0 Vanilla SL:** Rò rỉ 100% cấu trúc ảnh (SSIM ~0.995).
   - **B7 LightSplit (Fixed Orthogonal):** Nén rank $k=512, 1024, 2048$ vẫn rò rỉ hình thái vật thể (MLP SSIM $0.439 - 0.546$).
   - **B8 Adversarial (Full D):** Đánh bại được MLP nội bộ nhưng **vỡ trận hoàn toàn trước Fresh Conv Attacker** (Conv SSIM $0.6431$).
   - **AR-TAPE v1 (Đề xuất):** Nút thắt cổ chai trực giao kết hợp Min-Max Perceptual Obfuscation ($\lambda=0.25$, Mode F, Cosine) đạt:
     * Full Test Accuracy (10k): **$90.35 \pm 0.24\%$**
     * Fresh MLP SSIM: **$0.2159 \pm 0.0029$**
     * Fresh Conv SSIM: **$0.2014 \pm 0.0036$**
     * Active FSHA SSIM: **$0.0384$** (Task Acc duy trì **$84.25\%$**)
     * Tương quan $d\text{Cor}$: **$0.2876 \pm 0.0077$**

---

## 2. Hướng Dẫn Tái Lập (Replication Guide)

### Cách 1: Sử dụng Unified CLI Engine mới (Khuyến nghị)

Toàn bộ các thí nghiệm PoC có thể tái lập nhanh chóng qua các entrypoint chuẩn ở thư mục gốc:

1. **Huấn luyện AR-TAPE ($\lambda=0.25$, $k=2048$, $3$ seeds):**
   ```bash
   python train.py --dataset cifar10 --defense ar_tape --k 2048 --lambda-val 0.25 --scheduler cosine --epochs 100 --seed 42
   python train.py --dataset cifar10 --defense ar_tape --k 2048 --lambda-val 0.25 --scheduler cosine --epochs 100 --seed 7
   python train.py --dataset cifar10 --defense ar_tape --k 2048 --lambda-val 0.25 --scheduler cosine --epochs 100 --seed 2024
   ```

2. **Đánh giá Threat Surface (Fresh MLP & Conv):**
   ```bash
   python eval_attacks.py --client-ckpt output/cifar10_ar_tape_k2048_lam0.25_s42/checkpoints/cifar10_ar_tape_k2048_lam0.25_s42_client.pt --k 2048 --epochs 30
   ```

3. **Chạy Trọn Vẹn Benchmark & Xuất Bảng:**
   ```bash
   python benchmark.py --dataset cifar10 --defense ar_tape --k 2048 --lambda-val 0.25 --seeds 42 7 2024
   ```

---

### Cách 2: Sử dụng các Legacy Scripts nguyên bản (Historical Replication)

Các script gốc dùng trong quá trình nghiên cứu được lưu trữ tại `legacy_scripts/`:
- `run_step0_vanilla.py`: Baseline B0 Vanilla Split Learning.
- `run_step1_attack.py`: Inversion attack cơ bản trên B0.
- `run_step2_absorption.py`: Thí nghiệm chứng minh Server độc hại hấp thụ phép biến đổi cut-layer.
- `run_step3_baselines.py`: Triển khai các baseline B1-B5.
- `run_step3_b6_adp.py`: Baseline B6 (ADP-style AutoEncoder).
- `run_step3_b7_lightsplit.py`: Baseline B7 (Fixed Orthogonal Projection).
- `run_step4_fsha.py`: Tấn công chủ động FSHA trên các baseline.
- `train_artape_v0.py`: Script huấn luyện gốc của AR-TAPE (Mode F, min-max).
- `eval_artape_v0_fresh.py`: Script đánh giá Fresh MLP và Fresh Conv gốc.
- `run_artape_v0_fsha.py`: Script đánh giá FSHA trên client AR-TAPE.
- `run_artape_v1_utility_sweep.py`: Runner quét đa seed và utility sweep trên Google Colab.

---

## 3. Liên Kết Dữ Liệu & Báo Cáo
- Bảng Master Benchmark chính thức: [`master_benchmark_table.md`](../../master_benchmark_table.md)
- Báo cáo trực quan & phân tích: [`figures/pareto_frontier_tradeoff.png`](../../figures/pareto_frontier_tradeoff.png) và [`figures/visual_reconstruction_comparison.png`](../../figures/visual_reconstruction_comparison.png)
- Thông số chi tiết từng baseline: [`POC_RAW_DATA_AND_CONFIG.md`](POC_RAW_DATA_AND_CONFIG.md)
