# AR-TAPE: Absorption-Resistant Task-Aware Perceptual Encoding for Split Learning in Medical Image Analysis

> **Tài liệu đặc tả & Khung giàn thực nghiệm nghiên cứu (Research Harness & Specification)**

---

## 1. Bối cảnh & Động lực nghiên cứu (Context & Motivation)
* **Bối cảnh**: Chia sẻ dữ liệu liên viện (Collaborative Healthcare / Multi-site Split Learning) dưới ràng buộc bảo vệ quyền riêng tư nghiêm ngặt (HIPAA, GDPR).
* **Lỗ hổng nền tảng trong SL Training**:
  - Dữ liệu bị đập nát (*smashed data* $Z = F_{\text{client}}(X)$) gửi sang Server vẫn bảo toàn hình học và cấu trúc tri giác cấp cao.
  - Các kỹ thuật xáo trộn/mã hóa đối kháng hiện có (Xiao et al. AAAI 2020, DeepObfuscator 2019/2021, ADP 2025) chỉ được thiết kế cho giai đoạn *suy luận (inference)* với mô hình đóng băng.
  - Trong quá trình **huấn luyện SL**, Server cập nhật trọng số động $\theta_s$, dẫn tới **hiện tượng tự động hấp thụ ánh xạ ngược**: $\theta_s \to E^{-1}$ hoặc trích xuất khóa qua thống kê phân phối (Covariance Profiling). Đồng thời, xuất hiện thêm bề mặt tấn công chủ động **FSHA (Feature Space Hijacking Attack)**.

---

## 2. Khoảng trống tri thức & Đóng góp cốt lõi (Novelty & Contributions)

| Thành phần | Tình trạng trong Related Work | Vị trí Novelty của đề tài |
|---|---|---|
| **N1 — Phân tích & Mô hình hóa Hấp thụ $\theta_s \to E^{-1}$** | Chưa từng được phân tích trong SL training | **Đóng góp CV1**: Chứng minh toán học & thực nghiệm sự sụp đổ của phép mã hóa khả nghịch |
| **N2 — Encoder phi khả nghịch trong SL training** | Còn trống (hẹp) | **Đóng góp CV1**: Chiếu không gian con tác vụ $\mathbf{P}_{\text{task}} = \mathbf{V}\mathbf{V}^\top$, loại bỏ $\mathbf{P}_\perp z$ |
| **N3 — Đánh giá Thích ứng Thống nhất** | Đánh giá rời rạc, thiếu Kerckhoffs | **Đóng góp CV2**: Đánh giá 3 chiều (Utility, Security, Information Leakage) + FSHA stress-test |
| **CV3 — Kiểm chứng thực nghiệm Hai tầng** | Thiếu tính chuyển giao lâm sàng | Tầng 1: Controlled Verification (CIFAR) + Tầng 2: Application Case Study (HAM10000/PCAM) |

---

## 3. Câu hỏi nghiên cứu (Research Questions)
* **RQ1 (Thiết kế cơ chế)**: Cơ chế encoder phi khả nghịch tại cut layer — chiếu lên không gian con tác vụ $\mathbf{P}_{\text{task}} = \mathbf{V}_{\text{task}}\mathbf{V}_{\text{task}}^\top$ (tương đương loại bỏ thành phần $\mathbf{P}_\perp z$, với $\mathbf{P}_\perp = \mathbf{I} - \mathbf{P}_{\text{task}}$) — cần được thiết kế như thế nào để vừa duy trì đặc trưng bệnh học, vừa triệt tiêu tối đa thông tin cấu trúc tri giác?
* **RQ2 (Kháng hấp thụ & Bẻ gãy FSHA)**: Mô hình đề xuất ngăn chặn hiện tượng hấp thụ ánh xạ ngược ($\theta_s \to E^{-1}$) ở server, đồng thời chống lại các mạng sinh tái tạo thích ứng và tấn công chiếm dụng không gian đặc trưng (FSHA) hiệu quả ra sao, đo trên thang biến dạng thị giác đa mức độ (pixel-level và perceptual-level)?
* **RQ3 (Đánh đổi Bảo mật - Độ tin cậy lâm sàng)**: Sự đánh đổi giữa mức độ che giấu cấu trúc giải phẫu (PSNR, SSIM, LPIPS, Mutual Information) và độ tin cậy lâm sàng (AUC-ROC, F1-score, Grad-CAM attention alignment) của phương pháp đề xuất so với vanilla SL và các baseline B1–B6 đạt mức nào?

---

## 4. Kiến trúc Khung giàn Thử nghiệm (Two-Tier Research Harness)

```
                       [ Input Data: Image X ]
                                  │
                       [ Client Model F_c ]
                                  │
                      [ Raw Smashed Data Z ]
                                  │
    ┌─────────────────────────────┴─────────────────────────────┐
    │              AR-TAPE Cut-Layer Transformation             │
    │  1. Subspace Projection: P_task = V_task * V_task^T       │
    │     Eliminating: P_perp * z = (I - P_task) * z            │
    │  2. Non-linear irreversible quantization / activation     │
    └─────────────────────────────┬─────────────────────────────┘
                                  │ Encoded Z'
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[ Server Model F_s ]                             [ Threat Surface ]
  - Task Loss L_CE                                 - Passive Inversion (Decoder)
  - Split Learning Backprop                        - Kerckhoffs Adaptive Adapter
                                                   - Active FSHA Hijacking
                                                   - Information Leakage: dCor, MI
```

---

## 5. Danh mục Baselines So sánh (B0 – B6 & AR-TAPE)

1. **B0 (Vanilla SL)**: ResNet-18 phân tách tại Layer 1, không có cơ chế bảo vệ.
2. **B1 (Gaussian Noise)**: Bơm nhiễu Gauss đẳng hướng $Z' = Z + \mathcal{N}(0, \sigma^2 \mathbf{I})$.
3. **B2 (DP-SGD)**: Cắt xén gradient và cộng nhiễu Gauss trên Client Optimizer theo cơ chế Rényi Differential Privacy ($(\varepsilon, \delta)$-DP).
4. **B3 (NoPeek)**: Thêm hàm phạt khoảng cách tương quan $d\text{Cor}(X, Z')$ vào hàm mất mát tối ưu hóa: $\mathcal{L} = \mathcal{L}_{CE} + \alpha \cdot d\text{Cor}(X, Z')$.
5. **B4 (Block Scramble)**: Xáo trộn vị trí khối đặc trưng/pixel tại cut-layer bằng khóa bí mật.
6. **B5 (Deformable Operators)**: Biến dạng lưới tọa độ không gian phi tuyến ngẫu nhiên (Kiya et al. 2024).
7. **B6 (ADP-style AE)**: Chèn Autoencoder nút thắt cổ chai kết hợp nhiễu loạn phân phối (arXiv:2502.20629).
8. **B7 (LightSplit)**: Phép chiếu trực giao ngẫu nhiên cố định $R \in \mathbb{R}^{D \times k}$ ($R^\top R = I_k$, QR từ Gauss) tại cut-layer (arXiv:2605.13265, 05/2026). Baseline non-invertible duy nhất có $k \ll D$, chế độ F (lift không tham số) và L (MLP).
9. **Đề xuất: AR-TAPE**: Chiếu trực giao thu gọn rank nhận biết tác vụ $\mathbf{P}_{\text{task}} = \mathbf{V}_{\text{task}}\mathbf{V}_{\text{task}}^\top$, triệt tiêu không gian thừa $\mathbf{P}_\perp z$, phá vỡ toàn bộ quan hệ nghịch đảo của $\theta_s$.

---

## 6. Bộ Thước đo Đánh giá Thống nhất (Evaluation Metrics)

### A. Utility (Độ tin cậy tác vụ & Lâm sàng)
* **Tầng 1 (CIFAR-10 / CIFAR-100)**: Top-1 Test Accuracy.
* **Tầng 2 Nhị phân (PCAM)**:
  - AUC-ROC
  - Sensitivity & Specificity (ngưỡng 0.5 và ngưỡng tối ưu Youden's $J = \text{TPR} - \text{FPR}$)
  - Sensitivity tại ngưỡng $\text{Specificity} \ge 90\%$ (Sensitivity@90% Specificity)
* **Tầng 2 Đa lớp (HAM10000 - 7 lớp)**:
  - Macro-averaged AUC-ROC (One-vs-Rest)
  - Macro-averaged F1-score
  - Macro-averaged Sensitivity@90% Specificity
* **Visual Alignment**:
  - Grad-CAM attention alignment tại lớp tích chập cuối cùng: Hệ số tương quan Pearson ($r$) và Cosine Similarity giữa mô hình chuẩn và mô hình có mã hóa.

### B. Security (Mức độ sụp đổ tri giác của ảnh tái tạo)
* **PSNR (Peak Signal-to-Noise Ratio)**: Càng thấp càng tốt (mục tiêu $< 15\text{ dB}$).
* **SSIM (Structural Similarity Index)**: Càng thấp càng tốt (mục tiêu $< 0.40$).
* **LPIPS (Learned Perceptual Image Patch Similarity)**: Càng cao càng tốt (mục tiêu $> 0.50$).
* **MSE (Mean Squared Error)**: Lưu trữ ở phụ lục.

### C. Information Leakage (Rò rỉ thông tin)
* **Khoảng cách tương quan**: $d\text{Cor}(X, Z') \in [0, 1]$ (U-centering không thiên lệch).
* **Thông tin tương hỗ**: $I(X; Z')$ (ước lượng qua MINE / InfoNCE bound); mục tiêu triệt tiêu lượng thông tin thừa (*nuisance*): $\Delta I = I(X; Z') - I(Y; Z')$.

---

## 7. BẢNG CHECKLIST TIẾN ĐỘ THỰC HIỆN

- [x] **1. Thiết lập hạ tầng**: Môi trường `uv`, PyTorch 2.14, CUDA 13.0, GPU RTX 4060 Laptop.
- [x] **2. Bộ kiểm thử đơn vị tự động**: 9/9 modules đạt chuẩn 100% (`run_tests.py`).
- [x] **3. Tầng 1 — Bước 0**: Huấn luyện Vanilla Split Learning và Centralized đối chứng.
- [x] **4. Tầng 1 — Bước 1**: Tấn công giải mã tái tạo thụ động (Feature Inversion Decoder).
- [x] **5. Tầng 1 — Bước 2 (Novelty N1)**: Thực nghiệm chứng minh hiện tượng hấp thụ $\theta_s \to E^{-1}$ (Adapter match 87.5%, Covariance match 100%).
- [x] **6. Tầng 1 — Bước 3 (Baselines B1, B2, B3)**: Quét tham số và đánh giá đầy đủ toàn bộ metrics (PSNR, SSIM, LPIPS, $d\text{Cor}$).
- [x] **7. Cài đặt các Baselines mở rộng**:
  - [x] B4: Block Scramble (`src/defenses/block_scramble.py`)
  - [x] B5: Deformable Operator (`src/defenses/deformable.py`)
  - [x] B6: ADP-style Autoencoder (`src/defenses/adp_ae.py`)
- [x] **8. Cài đặt Cơ chế Đề xuất (Novelty N2)**:
  - [x] Module AR-TAPE Subspace Projection $\mathbf{P}_{\text{task}} = \mathbf{V}\mathbf{V}^\top$ (`src/defenses/ar_tape.py`)
- [x] **9. Cài đặt Bề mặt Tấn công Mở rộng**:
  - [x] Module stress-test FSHA (`src/attacks/fsha.py`)
- [x] **10. Cài đặt Bộ thước đo Mở rộng**:
  - [x] Module đo lường lâm sàng (AUC, Youden's J, Sens@90%Spec, Macro F1) (`src/metrics/clinical.py`)
  - [x] Module đo tương quan Grad-CAM Visual Alignment (`src/metrics/clinical.py`)
  - [x] Module đo lường Mutual Information & Leakage (`src/metrics/mutual_information.py`)
- [x] **11. Khung giàn điều phối & CLI Runner**:
  - [x] Pipeline tự động (`src/harness/pipeline.py`)
  - [x] CLI Runner tập trung (`run_harness.py`)
- [ ] **12. Huấn luyện & Đánh giá Tầng 1 trên AR-TAPE và B4–B6**:
  - [ ] Chạy huấn luyện và đánh giá AR-TAPE trên CIFAR-10.
  - [ ] So sánh đối đầu toàn diện B0–B6 vs AR-TAPE.
- [ ] **13. Tầng 2 — Thử nghiệm trên dữ liệu Y tế**:
  - [ ] Nạp dữ liệu HAM10000 hoặc PCAM.
  - [ ] Đánh giá Trade-off Privacy vs Clinical Utility & Grad-CAM Attention Alignment.

---

## 8. Hướng dẫn Lệnh điều khiển (CLI Command Guide)

| Tác vụ | Lệnh thực thi qua `uv` |
|---|---|
| **Xem Checklist tiến độ** | `uv run run_harness.py --checklist` |
| **Kiểm tra Smoke Test 9 modules** | `uv run run_harness.py --mode smoke_test` |
| **Chạy toàn bộ 9 Unit Tests** | `uv run run_tests.py` |
| **Hiển thị Bảng kết quả tổng hợp** | `uv run run_harness.py --mode eval_all` |
| **Huấn luyện Vanilla SL (B0)** | `uv run run_step0_vanilla.py --epochs 100` |
| **Huấn luyện Thí nghiệm Hấp thụ (Step 2)** | `uv run run_step2_absorption.py --mode adapter` |
| **Chạy Quét tham số Baselines (Step 3)** | `uv run run_step3_baselines.py --defense all` |
