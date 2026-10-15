# BÁO CÁO SỐ LIỆU THÔ VÀ CẤU HÌNH THỰC NGHIỆM PoC: B8 vs AR-TAPE v0
**Dự án:** AR-TAPE (Absorption-Resistant Task-Aware Perceptual Encoding) cho Split Learning  
**Ngày cập nhật & Đính chính:** 16/10/2026  
**Mục tiêu:** Đo đạc đối chứng Baseline B8 (min-max đơn thuần ở cut layer) và AR-TAPE v0 (ResSFL-lite: Bottleneck học được $k=1024$ kết hợp Adversary thích ứng mạnh ~34.4M tham số).

---

## 1. ĐÍNH CHÍNH CẤU HÌNH & LÀM RÕ CÁC CHỈ SỐ

### 1.1. Chuẩn hóa dữ liệu (Dataset Normalization)
- **Thông số thật trong mã nguồn (`src/data/cifar.py`):**
  - $\mu = (0.4914, 0.4822, 0.4465)$
  - $\sigma = (0.2023, 0.1994, 0.2010)$ (khớp $\sigma \approx 0.201$)
  *(Lưu ý: Bộ giá trị `0.247 / 0.243 / 0.262` ghi trong một số tài liệu cũ là nhầm lẫn; toàn bộ quá trình train/test đều dùng $\sigma \approx 0.201$).*

### 1.2. Bản chất của ngưỡng chặn $\text{cap} = 1.5586$
- Ngưỡng $\text{cap} = 1.5586$ là **kỳ vọng thực nghiệm $E[x_{norm}^2]$ đo trực tiếp trên tập ảnh CIFAR-10 sạch chuẩn hóa** (kết quả đo trong code: $\sim 1.5550$, lý thuyết phân phối $\approx 1.5586$).
- Đây chính là mức mất mát MSE của một bộ tái tạo tầm thường (trivial reconstructor luôn dự đoán hằng số 0, tức vector kỳ vọng $\mu$ của ảnh).
- **Không phải** là "MSE của phân phối chuẩn tắc $\mathcal{N}(0, 1)$" (vốn có $E[z^2] = 1.0$).

### 1.3. Đính chính Test Accuracy của Baseline B8
- **B8 ($\lambda=0.5$):** Full Test Acc (10.000 ảnh) = **89.84%** *(báo cáo cũ ghi nhầm 89.48%)*. Fast eval subset 2.000 ảnh tại epoch 100 đạt 90.60%, nhưng trung bình 10 epoch cuối chỉ đạt **69.35%** do mất ổn định huấn luyện min-max.
- **B8 ($\lambda=1.0$):** Full Test Acc (10.000 ảnh) = **89.48%**. Fast eval subset 2.000 ảnh tại epoch 100 đạt 91.60%, trung bình 10 epoch cuối đạt **91.04%**.

### 1.4. Phân định rõ 2k test vs 10k test và Giải thích độ lệch Acc của AR-TAPE v0
- **Số 87.34%:** Độ chính xác tại Epoch 100 đo trên **toàn bộ 10.000 ảnh test (Full Test Set)** ở chế độ `eval mode` (và ở chế độ `train mode` là 86.34%).
- **Số 89.78%:** Độ chính xác **trung bình 10 epoch cuối (Epochs 91 đến 100)** đo trên **tập con đánh giá nhanh 2.000 ảnh test cố định (Fast Eval Subset)** với seed 42.
- **Dữ liệu chi tiết 10 epoch cuối của AR-TAPE v0 ($\lambda=0.1, k=1024$):**
  - Epoch 91: 2k eval = 90.20% | 2k train = 85.95% | **10k Full Test = 87.04%**
  - Epoch 92: 2k eval = 89.55% | 2k train = 86.20% | **10k Full Test = 87.29%**
  - Epoch 93: 2k eval = 89.80% | 2k train = 86.75% | **10k Full Test = 87.37%**
  - Epoch 94: 2k eval = 90.05% | 2k train = 86.65% | **10k Full Test = 87.05%**
  - Epoch 95: 2k eval = 89.85% | 2k train = 86.55% | **10k Full Test = 87.31%**
  - Epoch 96: 2k eval = 89.70% | 2k train = 86.00% | **10k Full Test = 87.17%**
  - Epoch 97: 2k eval = 89.75% | 2k train = 86.40% | **10k Full Test = 87.26%**
  - Epoch 98: 2k eval = 89.50% | 2k train = 85.80% | **10k Full Test = 87.11%**
  - Epoch 99: 2k eval = 90.00% | 2k train = 86.20% | **10k Full Test = 87.24%**
  - Epoch 100: 2k eval = 89.35% | 2k train = 86.10% | **10k Full Test = 87.34%**
  $$\implies \text{Mean 10 epoch cuối trên 2k test} = \mathbf{89.78\%}$$
  $$\implies \text{Mean 10 epoch cuối trên Full 10k test} = \mathbf{87.22\%}$$
- **Nguyên nhân độ lệch:**
  1. *Lệch mẫu (Sampling Bias):* Tập 2.000 ảnh test ngẫu nhiên cố định (seed 42) có độ chính xác cao hơn tập 10.000 ảnh toàn thể khoảng $+2.5\%$.
  2. *Lệch thống kê BatchNorm:* Đoạn chênh giữa eval mode (87.34%) và train mode (86.34%) cho thấy thống kê running mean/variance của BatchNorm sau khi nén qua bottleneck $k=1024$ bị trôi nhẹ ~1.0%.

---

## 2. BẢNG TỔNG HỢP SỐ LIỆU THÔ PoC (B8 vs AR-TAPE v0)

| Thước đo / Chỉ số | B8 ($\lambda=0.1$) | B8 ($\lambda=0.5$) | B8 ($\lambda=1.0$) | **AR-TAPE v0 ($\lambda=0.1, k=1024$)** | Ngưỡng PoC Yêu Cầu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Test Acc (10k test, Ep 100)** | **92.38%** | **89.84%** | **89.48%** | **87.34%** *(86.34% train mode)* | $\ge 92.7\%$ |
| **Mean Test Acc (10 epoch cuối, 10k test)** | — | — | — | **87.22%** | $\ge 92.7\%$ |
| **Mean Test Acc (10 epoch cuối, 2k test)** | **93.42%** | 69.35% *(collapse)* | 91.04% | **89.78%** | Ổn định |
| **In-train Adv SSIM (eval mode)** | 0.4262 | 0.1527 | 0.1947 | **0.1239** | — |
| **In-train Adv SSIM (max)** | 0.4262 | 0.1531 | 0.1947 | **0.1239** | — |
| **In-train Adv PSNR (max)** | 14.31 dB | 12.43 dB | 12.10 dB | **12.41 dB** | — |
| **In-train Adv MSE** | 1.6212 | 1.7618 | 1.7381 | **1.6210** | — |
| **Distance Correlation ($d\text{Cor}$ Ep 100)**| 0.3928 | 0.2156 | 0.1945 | **0.3709** | Càng thấp càng tốt |
| **Thời gian train SL (s)** | 1361.2s | 1374.3s | 1382.3s | 2264.2s | — |
| --- | --- | --- | --- | --- | --- |
| **[Fresh MLP] SSIM (2k test)** | **0.4532** | **0.5947** | **0.4876** | **0.1859** | **< 0.30** |
| **[Fresh MLP] SSIM (10k test)** | 0.4580 | 0.5989 | 0.4908 | **0.1874** | **< 0.30** |
| **[Fresh MLP] PSNR (2k test)** | 17.92 dB | 19.42 dB | 18.48 dB | **13.89 dB** | **< 15.0 dB** |
| **[Fresh MLP] PSNR (10k test)** | 17.92 dB | 19.42 dB | 18.47 dB | **13.90 dB** | **< 15.0 dB** |
| **[Fresh MLP] LPIPS (2k)** | 0.1415 | 0.1111 | 0.1505 | **0.3333** | **> 0.20** |
| **[Fresh MLP] MSE Normalized** | 0.4764 | 0.3284 | 0.4115 | **1.1511** | $> 1.0$ |
| **Khoảng cách $G_{\text{MLP}}$** | $+0.0270$ | $\mathbf{+0.4416}$ | $\mathbf{+0.2929}$ | $\mathbf{+0.0620}$ | Triệt tiêu $\to 0$ |
| --- | --- | --- | --- | --- | --- |
| **[Fresh Conv] SSIM (2k test)** | **0.7641** | 0.1841 | **0.6552** | **0.1169** | **< 0.30** |
| **[Fresh Conv] SSIM (10k test)** | 0.7652 | 0.1843 | 0.6551 | **0.1172** | **< 0.30** |
| **[Fresh Conv] PSNR (2k test)** | 20.67 dB | 13.08 dB | 18.03 dB | **12.45 dB** | **< 15.0 dB** |
| **[Fresh Conv] PSNR (10k test)** | 20.63 dB | 13.06 dB | 18.00 dB | **12.44 dB** | **< 15.0 dB** |
| **[Fresh Conv] LPIPS (2k)** | 0.0237 | 0.1791 | 0.0465 | **0.3173** | **> 0.20** |
| **[Fresh Conv] MSE Normalized** | 0.2391 | 1.3467 | 0.4260 | **1.6508** | $> 1.0$ |
| **Khoảng cách $G_{\text{Conv}}$** | $\mathbf{+0.3379}$ | $+0.0310$ | $\mathbf{+0.4605}$ | $\mathbf{-0.0070}$ | Triệt tiêu $\to 0$ |

---

## 3. CHUỖI TIẾN HÓA $d\text{Cor}(X, Z)$ MỖI 10 EPOCHS

*(Đo lường sự phụ thuộc phi tuyến giữa ảnh gốc $X$ và smashed data $Z$ qua 100 epochs)*

| Epoch | B8 ($\lambda=0.1$) | B8 ($\lambda=0.5$) | B8 ($\lambda=1.0$) | **AR-TAPE v0 ($\lambda=0.1, k=1024$)** |
| :---: | :---: | :---: | :---: | :---: |
| **10** | 0.6057 | 0.5312 | 0.5670 | **0.4662** |
| **20** | 0.6416 | 0.6172 | 0.5528 | **0.4528** |
| **30** | 0.7354 | 0.6584 | 0.5858 | **0.4562** |
| **40** | 0.6506 | 0.6328 | 0.5769 | **0.4226** |
| **50** | 0.6626 | 0.7778 | 0.6667 | **0.4342** |
| **60** | 0.4457 | 0.3052 | 0.4344 | **0.4038** |
| **70** | 0.5608 | 0.3295 | 0.3662 | **0.3961** |
| **80** | 0.4041 | 0.2179 | 0.2572 | **0.3812** |
| **90** | 0.3728 | 0.2015 | 0.2083 | **0.3765** |
| **100**| **0.3928** | **0.2156** | **0.1945** | **0.3709** *(giảm đều, hội tụ mượt mà)* |

---

## 4. CẤU HÌNH THỰC NGHIỆM CHI TIẾT (HYPERPARAMETERS)

### 4.1. Dữ liệu huấn luyện & Đánh giá
- **Dataset:** CIFAR-10 ($32 \times 32 \times 3$).
- **Số lớp:** 10 lớp.
- **Phân chia:** Train = 50.000 ảnh, Full Test = 10.000 ảnh, Fast Eval Subset = 2.000 ảnh cố định (seed 42).
- **Chuẩn hóa thật:** $\text{mean} = [0.4914, 0.4822, 0.4465]$, $\text{std} = [0.2023, 0.1994, 0.2010]$.

### 4.2. Cấu trúc mạng AR-TAPE v0
- **Client ($E_\psi$):**
  - ResNet-18 Backbone: `stem` (`Conv2d(3, 64, 3, 1, 1)`, `BatchNorm2d`, `ReLU`) + `Layer 1` (2 BasicBlock, 64 channels).
  - Kích thước feature map tại cut layer: $64 \times 32 \times 32 = 65.536$ chiều phẳng ($D = 65.536$).
  - **Learned Bottleneck:** `nn.Linear(65536, 1024, bias=False)`.
  - Khởi tạo ma trận nén: **QR Orthonormalization** trên phân phối chuẩn ngẫu nhiên (seed 42).
  - Tỉ lệ nén: **64:1**.
- **Server:**
  - **Projection MLP:** `Linear(1024, 512)` $\to$ `BatchNorm1d(512)` $\to$ `ReLU` $\to$ `Linear(512, 65536)` $\to$ `BatchNorm1d(65536)`. Reshape lại thành tensor $(B, 64, 32, 32)$.
  - ResNet-18 Backbone: `Layer 2` (128 channels), `Layer 3` (256 channels), `Layer 4` (512 channels) + `AdaptiveAvgPool2d((1, 1))` + `Linear(512, 10)`.
- **In-training Adversary (LearnedAdaptiveDecoder):**
  - Quy mô: **~34.4 triệu tham số**.
  - Unprojector MLP: `Linear(1024, 512)` $\to$ `BatchNorm1d(512)` $\to$ `ReLU` $\to$ `Linear(512, 65536)` $\to$ `BatchNorm1d(65536)`.
  - Conv Decoder: 4 tầng `ConvTranspose2d` ($64 \to 64 \to 32 \to 16 \to 3$) với BatchNorm và ReLU.

### 4.3. Tối ưu hóa & Huấn luyện Min-Max (Split Learning)
- **Số Epochs:** 100 epochs.
- **Batch Size:** 128.
- **Client & Server Optimizer:** SGD ($\text{lr} = 0.1$, $\text{momentum} = 0.9$, $\text{weight\_decay} = 5 \times 10^{-4}$).
  - Scheduler: `MultiStepLR` (milestones = [50, 75], gamma = 0.1).
- **Adversary Optimizer:** Adam ($\text{lr} = 1 \times 10^{-3}$, $\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = 0).
- **Cơ chế Chặn MSE Loss (Cap):**
  - $\text{cap} = 1.5586$ (tương ứng $E[x_{norm}^2]$ thực nghiệm đo trên ảnh sạch chuẩn hóa).
- **Vòng lặp Min-Max mỗi batch:**
  1. $z = \text{Client}(x) \in \mathbb{R}^{B \times 1024}$.
  2. **Bước Max (Cập nhật Adversary):**
     $$\mathcal{L}_{adv} = \text{MSE}(\text{Adv}(z.\text{detach}()), x_{norm})$$
  3. **Bước Min (Cập nhật Client & Server):**
     $$\hat{y} = \text{Server}(z)$$
     $$\mathcal{L}_{task} = \text{CrossEntropy}(\hat{y}, y)$$
     $$\mathcal{L}_{rec} = \text{clamp}\big(\text{MSE}(\text{Adv}(z), x_{norm}), \; \text{max}=1.5586\big)$$
     $$\mathcal{L}_{total} = \mathcal{L}_{task} - \lambda \cdot \mathcal{L}_{rec} \quad (\lambda = 0.1)$$
     *(Adversary bị đóng băng ở bước Min, chỉ lan truyền gradient về Client và Server).*

### 4.4. Kiểm thử Fresh Attackers (Nguyên tắc Kerckhoffs)
- **Điều kiện thử nghiệm:** Client đóng băng hoàn toàn (`requires_grad=False`, chế độ `eval()`). Đối thủ khởi tạo ngẫu nhiên hoàn toàn mới và huấn luyện trên toàn bộ 50.000 ảnh CIFAR-10.
- **Fresh Learned Adaptive MLP Attacker:**
  - Cấu trúc: `Unprojector (1024 -> 512 -> 65536)` + `ConvDecoder` (~34.4M tham số).
  - Tối ưu: Adam $\text{lr} = 1 \times 10^{-3}$, `CosineAnnealingLR` (T_max = 30), 30 epochs, batch size 128.
- **Fresh Conv Decoder Attacker:**
  - Cấu trúc: `Linear(1024, 65536)` + 4 tầng `ConvTranspose2d` (~5.4M tham số).
  - Tối ưu: Adam $\text{lr} = 1 \times 10^{-3}$, `CosineAnnealingLR` (T_max = 30), 30 epochs, batch size 128.

---

## 5. KẾ HOẠCH HÀNH ĐỘNG CHO 4 HẠNG MỤC CÒN THIẾU

1. **Chạy đối chứng $\lambda=0$ (Ablation thuần túy Bottleneck $k=1024$):**
   - Huấn luyện không có adversary.
   - Đo Fresh MLP & Fresh Conv trên Client $\lambda=0$ để tách rời đóng góp của bottleneck (nén 64:1) và của min-max adversary.
2. **Chạy 2 Attacker còn thiếu trên Client v0:**
   - **FSHA ($g_s = 5$ và $g_s = 1$, 30 epochs):** Đo acc hội tụ, PSNR và SSIM ở đỉnh (peak) và ở cuối (final).
   - **Co-adapted nhẹ với $N_{aux} = 10.000$:** Đo độ chênh $\Delta = \text{SSIM}_{\text{last10}}(A) - \text{SSIM}_{\text{last10}}(B)$.
3. **Cải thiện Utility (Tiến tới ngưỡng $\ge 92.7\%$):**
   - Thử nghiệm $k = 2048$ (nén 32:1).
   - Thử nghiệm warmup $\lambda$ (10 epoch đầu) và kiểm soát trôi thống kê BatchNorm.
4. **Mở rộng Đa Seed:**
   - Chạy Seed 7 và Seed 2024 sau khi xác định điểm vận hành tối ưu từ các bước trên.
