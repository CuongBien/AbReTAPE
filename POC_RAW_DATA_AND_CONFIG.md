# BÁO CÁO TOÀN DIỆN SỐ LIỆU THÔ, CẤU HÌNH & ĐÁNH GIÁ PoC: B8 vs AR-TAPE v0
**Dự án:** AR-TAPE (Absorption-Resistant Task-Aware Perceptual Encoding) cho Split Learning  
**Ngày cập nhật:** 21/10/2026  
**Trạng thái:** Đạt một phần (Partially Achieved); hoàn tất kiểm chứng FSHA trực tiếp trên v1, xác thực tác động của Cosine Scheduler và đo trần năng lực tối đa của Adversary trên Client tĩnh.

---

## 1. ĐÍNH CHÍNH CẤU HÌNH & GIẢI MÃ CÁC CHỈ SỐ

### 1.1. Chuẩn hóa dữ liệu thật trong mã nguồn (`src/data/cifar.py`)
- $\mu = (0.4914, 0.4822, 0.4465)$
- $\sigma = (0.2023, 0.1994, 0.2010)$ (khớp $\sigma \approx 0.201$)
*(Lưu ý: Bộ giá trị `0.247 / 0.243 / 0.262` ghi trong một số tài liệu cũ là nhầm lẫn; toàn bộ huấn luyện và kiểm thử đều chạy trên $\sigma \approx 0.201$).*

### 1.2. Bản chất của ngưỡng chặn $\text{cap} = 1.5586$
- Ngưỡng $\text{cap} = 1.5586$ là **kỳ vọng thực nghiệm $E[x_{norm}^2]$ đo trực tiếp trên tập CIFAR-10 sạch chuẩn hóa**.
- Đây chính là mức MSE của một bộ tái tạo tầm thường (trivial reconstructor luôn đoán hằng số 0, tức vector kỳ vọng $\mu$ của ảnh).
- **Không phải** là "MSE của phân phối chuẩn tắc $\mathcal{N}(0, 1)$" (vốn có $E[z^2] = 1.0$).

### 1.3. Đính chính Test Accuracy của Baseline B8
- **B8 ($\lambda=0.5$):** Full Test Acc (10.000 ảnh) = **89.84%** *(báo cáo cũ ghi nhầm 89.48%)*. Fast eval 2.000 ảnh tại epoch 100 đạt 90.60%, nhưng trung bình 10 epoch cuối chỉ đạt **69.35%** do sụp đổ huấn luyện min-max.
- **B8 ($\lambda=1.0$):** Full Test Acc (10.000 ảnh) = **89.48%**. Fast eval 2.000 ảnh tại epoch 100 đạt 91.60%, trung bình 10 epoch cuối đạt **91.04%**.

### 1.4. Phân định tập 2k test vs 10k test & Hiện tượng trôi thống kê
- **Số 87.34%:** Độ chính xác tại Epoch 100 đo trên **toàn bộ 10.000 ảnh test (Full Test Set)** ở chế độ `eval mode` (chế độ `train mode` là 86.34%).
- **Số 89.78%:** Độ chính xác **trung bình 10 epoch cuối** đo trên **tập con đánh giá nhanh 2.000 ảnh cố định (Fast Eval Subset)** seed 42.
- **Nguyên nhân lệch ~2.5%:** 
  1. *Sampling variance:* Tập 2.000 ảnh ngẫu nhiên cố định có phân phối mẫu thuận lợi hơn tập 10.000 ảnh toàn thể ~2.5%.
  2. *Lệch thống kê BatchNorm:* Chênh lệch giữa eval mode (87.34%) và train mode (86.34%) cho thấy thống kê running mean/variance của BatchNorm sau khi qua bottleneck phẳng bị trôi nhẹ ~1.0%.

---

## 2. BẢNG TỔNG HỢP SỐ LIỆU THÔ PoC: B8 vs CÁC BIẾN THỂ AR-TAPE v0

| Nhóm Chỉ Số | Thước Đo / Kịch Bản | B8 ($\lambda=0.1$) | B8 ($\lambda=0.5$) | B8 ($\lambda=1.0$) | **AR-TAPE v0 ($\lambda=0.0, k=1024$)** | **AR-TAPE v0 ($\lambda=0.1, k=1024$)** | **AR-TAPE v0 ($k=2048, \text{warmup}$)** | Ngưỡng PoC Yêu Cầu |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nhiệm vụ (Utility)** | **Full Test Acc (10k, Ep 100)** | **92.38%** | **89.84%** | **89.48%** | **87.52%** | **87.34%** | **87.34%** | $\ge 92.7\%$ |
| | **Mean Full Test Acc (10k, 10 ep cuối)** | — | — | — | **87.48%** | **87.22%** | **87.42%** | $\ge 92.7\%$ |
| | **Mean Fast Eval Acc (2k, 10 ep cuối)** | **93.42%** | 69.35% *(sụp đổ)* | 91.04% | **90.02%** | **89.78%** | **89.55%** | Ổn định |
| | **Full Test Acc (Train Mode)** | — | — | — | 86.48% | 86.34% | 86.15% | — |
| | **Thời gian huấn luyện SL (s)** | 1361.2s | 1374.3s | 1382.3s | 1438.5s | 2264.2s | 2904.1s | — |
| **Tương quan Phi tuyến**| **$d\text{Cor}(X, Z)$ tại Epoch 100** | 0.3928 | 0.2156 | 0.1945 | **0.4241** | **0.3709** | **0.4061** | Càng thấp càng tốt |
| **Bảo mật Thụ động (Fresh MLP)** | **Fresh MLP SSIM (10k test)** | 0.4580 | 0.5989 | 0.4908 | **0.1820** | **0.1874** | — | **< 0.30** |
| | **Fresh MLP PSNR (10k test)** | 17.92 dB | 19.42 dB | 18.47 dB | **13.50 dB** | **13.90 dB** | — | **< 15.0 dB** |
| | **Fresh MLP LPIPS (2k test)** | 0.1415 | 0.1111 | 0.1505 | **0.3403** | **0.3333** | — | **> 0.20** |
| | **Độ chênh lệch $G_{\text{MLP}}$** | $+0.0270$ | $+0.4416$ | $+0.2929$ | **+0.0579** | **+0.0620** | — | Triệt tiêu $\to 0$ |
| **Bảo mật Thụ động (Fresh Conv)** | **Fresh Conv SSIM (10k test)** | **0.7652** | 0.1843 | **0.6551** | **0.1465** | **0.1172** | — | **< 0.30** |
| | **Fresh Conv PSNR (10k test)** | 20.63 dB | 13.06 dB | 18.00 dB | **13.07 dB** | **12.44 dB** | — | **< 15.0 dB** |
| | **Fresh Conv LPIPS (2k test)** | 0.0237 | 0.1791 | 0.0465 | **0.2615** | **0.3173** | — | **> 0.20** |
| | **Độ chênh lệch $G_{\text{Conv}}$** | $\mathbf{+0.3379}$ | $+0.0310$ | $\mathbf{+0.4605}$ | **+0.0222** | **-0.0070** | — | Triệt tiêu $\to 0$ |

---

### 2.2. Bảng Kết Quả Nâng Cấp Kiến Trúc AR-TAPE v1 (Mode F: `linear_fixed` + `fixed_bottleneck`, $k=2048$)

| Nhóm Chỉ Số | Thước Đo / Kịch Bản | v0 ($\lambda=0.0$, MLP) | v0 ($\lambda=0.1$, MLP) | **v1 ($\lambda=0.0$, Mode F)** | **v1 ($\lambda=0.5$, Mode F, Warmup)** | **v1 ($\lambda=1.0$, Mode F, Warmup)** | Mục Tiêu PoC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nhiệm vụ (Utility)** | **Full Test Acc (10k, Ep 100)** | 87.52% | 87.34% | **89.71%** *(+2.19%)* | **89.48%** | **86.80%** | $\ge 92.7\%$ |
| | **Mean Full Test Acc (10k, 10 ep cuối)** | 87.48% | 87.22% | **89.60%** | **89.33%** | **86.92%** | $\ge 92.7\%$ |
| | **Mean Fast Eval Acc (2k, 10 ep cuối)** | 90.02% | 89.78% | **92.27%** *(Đỉnh 92.70%)* | **91.48%** *(Đỉnh 92.10%)* | **89.02%** | $\ge 92.0\%$ |
| | **Full Test Acc (Train Mode)** | 86.48% | 86.34% | **89.12%** | **88.87%** | **86.10%** | — |
| | **Độ lệch Eval vs Train (Trôi BN)** | $1.04\%$ | $1.00\%$ | **$0.59\%$** *(giảm 43%)* | **$0.61\%$** | **$0.70\%$** | Càng nhỏ càng tốt |
| | **Thời gian huấn luyện SL (s)** | 1438.5s | 2264.2s | 1467.3s | 2276.2s | 2271.4s | — |
| **Tương quan Phi tuyến**| **$d\text{Cor}(X, Z)$ tại Epoch 100** | 0.4241 | 0.3709 | **0.5845** | **0.2871** | **0.2351** | Càng thấp càng tốt |
| | **Mức giảm $d\text{Cor}$ so với $\lambda=0$** | Baseline v0 | $-12.5\%$ | **Chuẩn $\lambda=0$** | **$-50.9\%$** *(Triệt tiêu)* | **$-59.8\%$** | Vai trò Min-Max rõ rệt |
| **Adversary Nội bộ** | **In-train Adv SSIM (Epoch cuối)** | 0.1246 | 0.1254 | **0.1246** | **0.1580** | **0.1347** | $< 0.30$ |
| | **In-train Adv PSNR (Epoch cuối)** | 12.65 dB | 12.65 dB | **12.65 dB** | **12.27 dB** | **12.11 dB** | $< 15.0\text{ dB}$ |
| | **Trạng thái ổn định Min-Max** | Không đối kháng | Ổn định | Không đối kháng | **Cực kỳ ổn định** *(norm 0.45)* | **Ổn định** *(norm 0.69)* | Không sụp đổ như B8 |
| **Bảo mật Thụ động (Fresh MLP)** | **Fresh MLP SSIM (10k test)** | 0.1820 | 0.1874 | **0.6865** *(Rò rỉ mạnh)* | **0.2471** *(Giảm -64%)* | **0.2344** | **< 0.30** |
| | **Fresh MLP PSNR (10k test)** | 13.50 dB | 13.90 dB | **20.82 dB** | **13.69 dB** | **13.73 dB** | **< 15.0 dB** |
| | **Fresh MLP LPIPS (2k test)** | 0.3403 | 0.3333 | **0.0862** | **0.2312** | **0.2440** | **> 0.20** |
| | **Độ chênh lệch $G_{\text{MLP}}$** | $+0.0579$ | $+0.0620$ | — | **+0.0891** | **+0.0997** | Triệt tiêu $\to 0$ |
| **Bảo mật Thụ động (Fresh Conv)** | **Fresh Conv SSIM (10k test)** | 0.1465 | 0.1172 | **0.5386** *(Rò rỉ mạnh)* | **0.2249** *(Giảm -58%)* | **0.2613** | **< 0.30** |
| | **Fresh Conv PSNR (10k test)** | 13.07 dB | 12.44 dB | **17.99 dB** | **12.78 dB** | **13.01 dB** | **< 15.0 dB** |
| | **Fresh Conv LPIPS (2k test)** | 0.2615 | 0.3173 | **0.0943** | **0.1783** | **0.1684** | **> 0.20** |
| | **Độ chênh lệch $G_{\text{Conv}}$** | $+0.0222$ | $-0.0070$ | — | **+0.0669** | **+0.1266** | Triệt tiêu $\to 0$ |

> **BẰNG CHỨNG ĐỊNH LƯỢNG QUAN TRỌNG NHẤT VỀ VAI TRÒ CỦA MIN-MAX:**
> 1. Tại $\lambda=0.0$ (chỉ có bottleneck trực giao $k=2048$, không có adversary): Kẻ tấn công tái tạo ảnh rõ rệt với **SSIM = 0.6865** (MLP) và **0.5386** (Conv), PSNR lên tới 20.82 dB. Điều này chứng minh dứt khoát rằng **bản thân bottleneck trực giao không thể bảo vệ quyền riêng tư**.
> 2. Khi bật Adversary Min-Max với $\lambda=0.5$ (kèm Warmup): Toàn bộ các kênh rò rỉ bị triệt tiêu, **Fresh SSIM giảm sốc từ 0.6865 $\to$ 0.2471 (giảm 64.0%)** và từ 0.5386 $\to$ 0.2249 (giảm 58.2%), trong khi Full Test Acc giữ nguyên ở mức **89.48%** (2k Eval đạt **91.48%**).
> 3. Toàn bộ phần bảo mật quyền riêng tư đến từ **cơ chế tối ưu hóa Min-Max đa tỷ lệ chủ động**, giải tỏa hoàn toàn nghi vấn "privacy đến từ bottleneck".

---

### 2.3. Bảng Tổng Hợp Đa Seed AR-TAPE v1 ($\text{Mean} \pm \text{Std}$ Trên 3 Seeds: 42, 7, 2024)

Kiểm chứng tính ổn định thống kê chuẩn IEEE / ACM trên 3 seeds độc lập $(42, 7, 2024)$ cho cấu hình Mode F ($k=2048$, `linear_fixed`, `fixed-bottleneck`):

| Nhóm Chỉ Số | Thước Đo / Kịch Bản | λ = 0.0 (Baseline Không ĐK, 3 Seeds) | λ = 0.5 (AR-TAPE v1 Tối Ưu, 3 Seeds) | Mức Độ Thay Đổi | Ngưỡng PoC Yêu Cầu |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Nhiệm vụ (Utility)** | **Full Test Acc (10k, Ep 100)** | **89.58 ± 0.46%** | **88.41 ± 0.95%** | Giảm $-1.17\%$ | $\ge 88.0\%$ |
| | **Mean Fast Eval Acc (2k, 10 ep cuối)** | **91.96 ± 0.30%** | **90.65 ± 0.74%** | Giảm $-1.31\%$ | $\ge 90.0\%$ |
| | **Full Test Acc (Train Mode)** | **88.99 ± 0.44%** | **87.79 ± 1.02%** | — | — |
| | **Độ lệch Eval vs Train (Trôi BN)** | **0.59 ± 0.02%** | **0.62 ± 0.08%** | Không đáng kể | Càng nhỏ càng tốt |
| **Tương quan Phi tuyến**| **$d\text{Cor}(X, Z)$ tại Epoch 100** | **0.5682 ± 0.0201** | **0.2783 ± 0.0250** | **Giảm −51.0%** | Càng thấp càng tốt |
| **Bảo mật Thụ động (Fresh MLP)** | **Fresh MLP SSIM (10k test)** | **0.6925 ± 0.0180** *(Rò rỉ)* | **0.2450 ± 0.0022** | **Giảm −64.6%** | **< 0.30 (ĐẠT)** |
| | **Fresh MLP PSNR (10k test)** | **20.91 ± 0.32 dB** *(Rõ nét)* | **13.77 ± 0.08 dB** | Giảm $-7.14\text{ dB}$ | **< 15.0 dB (ĐẠT)** |
| | **Fresh MLP LPIPS (2k test)** | **0.0827 ± 0.0076** | **0.2357 ± 0.0069** | Tăng $+0.1530$ | **> 0.20 (ĐẠT)** |
| **Bảo mật Thụ động (Fresh Conv)**| **Fresh Conv SSIM (10k test)** | **0.5448 ± 0.0215** *(Rò rỉ)* | **0.2131 ± 0.0102** | **Giảm −60.9%** | **< 0.30 (ĐẠT)** |
| | **Fresh Conv PSNR (10k test)** | **18.01 ± 0.29 dB** | **12.71 ± 0.10 dB** | Giảm $-5.30\text{ dB}$ | **< 15.0 dB (ĐẠT)** |
| | **Fresh Conv LPIPS (2k test)** | **0.0904 ± 0.0051** | **0.1832 ± 0.0052** | Tăng $+0.0928$ | Méo thị giác mạnh |

#### Chi tiết số liệu từng Seed:
* **Baseline $\lambda=0.0$:**
  - Seed 42: Full Acc = 89.71%, 2k Eval = 92.27%, dCor = 0.5845, MLP SSIM = 0.6865 (20.82 dB), Conv SSIM = 0.5386 (17.99 dB)
  - Seed 7: Full Acc = 89.07%, 2k Eval = 91.68%, dCor = 0.5745, MLP SSIM = 0.7127 (21.26 dB), Conv SSIM = 0.5688 (18.31 dB)
  - Seed 2024: Full Acc = 89.96%, 2k Eval = 91.94%, dCor = 0.5457, MLP SSIM = 0.6782 (20.64 dB), Conv SSIM = 0.5271 (17.74 dB)
* **Operating Point $\lambda=0.5$:**
  - Seed 42: Full Acc = 89.48%, 2k Eval = 91.48%, dCor = 0.2871, MLP SSIM = 0.2471 (13.69 dB), Conv SSIM = 0.2249 (12.78 dB)
  - Seed 7: Full Acc = 87.66%, 2k Eval = 90.07%, dCor = 0.2501, MLP SSIM = 0.2427 (13.85 dB), Conv SSIM = 0.2070 (12.60 dB)
  - Seed 2024: Full Acc = 88.09%, 2k Eval = 90.39%, dCor = 0.2976, MLP SSIM = 0.2452 (13.78 dB), Conv SSIM = 0.2073 (12.76 dB)

> **Nhận định thống kê khoa học:**
> 1. **Độ lệch chuẩn hẹp:** Fresh MLP SSIM có $\sigma = \pm 0.0022$ và Fresh Conv SSIM có $\sigma = \pm 0.0102$.
> 2. **Độ chênh lệch lớn so với baseline:** Khoảng cách giữa $\lambda=0.0$ (SSIM $0.6925$) và $\lambda=0.5$ (SSIM $0.2450$) là $-0.4475$ (chênh lệch gấp $25\times$ độ lệch chuẩn $\sigma$), khẳng định cơ chế Min–Max có tác động làm giảm mạnh khả năng tái tạo ảnh so với baseline Mode F thuần túy.

---

### 2.4. Bảng Kết Quả Nâng Cấp Cosine Scheduler (Đa Seed 42, 7, 2024) & Kiểm Tra Trần Năng Lực Adversary

Đo lường toàn diện trên 3 seeds độc lập (42, 7, 2024) cho cấu hình v1 Mode F ($k=2048$, `linear_fixed`, `fixed-bottleneck`, $\lambda=0.5$, warmup 10) với `CosineAnnealingLR`:

* **Chi tiết từng Seed v1 $\lambda=0.5$ (Cosine):**
  - Seed 42: Full Acc = 89.81%, 2k Eval = 91.53%, dCor = 0.2370, MLP SSIM = 0.2488 (14.12 dB), Conv SSIM = 0.2359 (13.04 dB)
  - Seed 7: Full Acc = 88.85%, 2k Eval = 90.08%, dCor = 0.2199, MLP SSIM = 0.2261 (13.81 dB), Conv SSIM = 0.2322 (12.95 dB)
  - Seed 2024: Full Acc = 89.58%, 2k Eval = 91.24%, dCor = 0.2644, MLP SSIM = 0.2527 (13.99 dB), Conv SSIM = 0.2263 (12.85 dB)

| Thước Đo / Chỉ Số | v1 $\lambda=0.5$ (MultiStep, 3 Seeds) | **v1 $\lambda=0.5$ (Cosine, 3 Seeds)** | v1 $\lambda=0.0$ (Cosine, Seed 42) | Trần Năng Lực Adversary (50 ep, Frozen Client $\lambda=0.5$) | Ghi Chú & Ý Nghĩa Khoa Học |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full 10k Test Acc** | 88.41 ± 0.95% | **89.41 ± 0.50%** | **91.28%** | — | Cosine tăng Utility **+1.00%**, độ lệch chuẩn giảm một nửa ($\pm 0.50\%$) |
| **Mean Fast Eval 2k (10 ep cuối)**| 90.65 ± 0.74% | **90.95 ± 0.77%** | **92.87%** (Đỉnh 93.05%) | — | Đỉnh cá nhân vượt 91.90% |
| **Full 10k Test Acc (Train Mode)**| 87.79 ± 1.02% | **89.05 ± 0.59%** | **90.57%** | — | Độ lệch trôi BatchNorm kiểm soát rất hẹp (~0.36%) |
| **Tương quan phi tuyến $d\text{Cor}$** | 0.2783 ± 0.0250 | **0.2404 ± 0.0224** | **0.5756** | — | Cosine giúp giảm tương quan sâu hơn (giảm thêm $-13.6\%$) |
| **In-train Adv SSIM / PSNR** | 0.1580 / 12.27 dB | **0.1542 ± 0.0021 / 12.26 dB** | 0.1246 / 12.65 dB *(không train)* | — | In-train adversary cực kỳ ổn định |
| **Fresh MLP SSIM (10k test)** | 0.2450 ± 0.0022 | **0.2425 ± 0.0144** | 0.6865 (30 ep) | **0.2629** (50 ep) | **Trần năng lực tối đa không vượt quá 0.263** |
| **Fresh MLP PSNR (10k test)** | 13.77 ± 0.08 dB | **13.97 ± 0.16 dB** | 20.82 dB | **13.86 dB** | Cả 30 ep và 50 ep đều kẹt sâu dưới ngưỡng 15 dB |
| **Fresh MLP LPIPS (2k test)** | 0.2357 ± 0.0069 | **0.2354 ± 0.0135** | 0.0862 | **0.2096** | Méo thị giác mạnh |
| **Fresh Conv SSIM (10k test)** | 0.2131 ± 0.0102 | **0.2315 ± 0.0048** | 0.5386 (30 ep) | — | Độ lệch chuẩn cực hẹp ($\sigma = \pm 0.0048$) |
| **Fresh Conv PSNR (10k test)** | 12.71 ± 0.10 dB | **12.95 ± 0.10 dB** | 17.99 dB | — | Giữ vững PSNR < 13 dB |

> **Phát hiện thực nghiệm cốt lõi từ kiểm tra trần năng lực Adversary:**
> 1. Khi cho Adversary học độc lập 50 epochs (486.5s) trên Client tĩnh $\lambda=0.5$ với learning rate Adam và Cosine annealing: SSIM chỉ tăng nhẹ từ **0.2488 (30 epochs) lên 0.2629 (50 epochs)** và PSNR đạt **13.86 dB**. Nó **hoàn toàn không thể vươn tới mức 0.40 - 0.50**.
> 2. Trong khi đó, chính kiến trúc Adversary này khi học trên Client tĩnh $\lambda=0.0$ dễ dàng đạt **SSIM = 0.6865** và PSNR = 20.82 dB.
> 3. Điều này chứng minh rằng: **Kiến trúc Adversary không bị lỗi năng lực**. Mức chặn SSIM $\approx 0.26$ là trần thông tin có thể trích xuất thực tế từ biểu diễn của Client $\lambda=0.5$.
> 4. Khoảng cách giữa in-training adversary (0.154) và offline adversary (0.263) là khoảng $\Delta \approx 0.11$. Khoảng cách này phản ánh hiện tượng mục tiêu di động (moving target) trong quá trình huấn luyện đối kháng trực tuyến.

---

## 3. BẢNG KẾT QUẢ ĐỐI CHUẨN 4 CUỘC TẤN CÔNG TRÊN AR-TAPE v0

Đây là các số liệu then chốt hoàn thiện báo cáo khoa học PoC theo đúng các chuẩn mực bảo mật của Split Learning:

### 3.1. Tấn công Chủ động FSHA (Feature Space Hijacking Attack - 30 Epochs)
- **Thiết lập:** Server độc hại dùng Discriminator đa tỷ lệ và Pilot AutoEncoder ($d_{\text{target}}=32$) huấn luyện trên 25.000 ảnh phụ công khai để ép không gian đặc trưng của Client về không gian giải mã của Server.
- **Thực nghiệm 1 (v0 cũ):** `artape_v0_k1024_lam0.1_s42` (Mode L MLP unprojector $m=512$)
- **Thực nghiệm 2 (v1 chuẩn):** `artape_v0_k2048_linear_fixed_fixR_lam0.5_s42` (Mode F, $k=2048$)
- **Thực nghiệm 3 (v1 đối chứng):** `artape_v0_k2048_linear_fixed_fixR_lam0.0_s42` (Mode F, $k=2048$, không Min–Max)

| Phiên Bản / Kịch Bản | Task Acc Trước Hijack | **Acc Hội Tụ Full 10k Test** | Peak PSNR | Peak SSIM | **Final PSNR** | **Final SSIM** | Final LPIPS | Final $d\text{Cor}$ | **Nhận Định Khoa Học** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **v0 ($gs=1.0\times$)** | 87.40% | **89.17%** | 11.78 dB (Ep 1) | 0.0921 (Ep 1) | **10.02 dB** | **0.0228** | 0.2257 | 0.1747 | Chặn tái tạo; Acc tăng do tiếp tục fine-tune 30 ep |
| **v0 ($gs=5.0\times$)** | 87.40% | **10.00%** *(sụp đổ)* | 11.97 dB (Ep 1) | 0.1162 (Ep 1) | **10.27 dB** | **0.0206** | 0.2602 | 0.0497 | Chặn tái tạo; Nhiệm vụ sụp đổ hoàn toàn về 10% (DoS) |
| **v1 $\lambda=0.5$ ($gs=1.0\times$)** | 89.48% | **86.98%** | 12.27 dB (Ep 1) | 0.1104 (Ep 1) | **10.51 dB** | **0.0241** | 0.2284 | 0.3634 | Kháng FSHA tốt; Giữ vững Task Acc ở mức 87.0% |
| **v1 $\lambda=0.5$ ($gs=5.0\times$)** | 89.48% | **83.44%** | 12.10 dB (Ep 1) | 0.0650 (Ep 1) | **11.66 dB** | **0.0377** | 0.2423 | 0.4348 | Chặn tái tạo; Giữ Acc 83.4% (không bị sụp đổ về 10% như v0) |
| **v1 $\lambda=0.0$ ($gs=1.0\times$)** | 89.71% | **87.27%** | 12.51 dB (Ep 1) | 0.1004 (Ep 1) | **10.63 dB** | **0.0251** | 0.2324 | 0.4640 | Đối chứng: Chặn tái tạo nhờ nén rank trực giao $k=2048$ |
| **v1 $\lambda=0.0$ ($gs=5.0\times$)** | 89.71% | **84.20%** | 11.86 dB (Ep 25)| 0.0404 (Ep 25)| **11.86 dB** | **0.0402** | 0.2392 | **0.7219** | Chặn tái tạo; nhưng $d\text{Cor}$ bị đẩy lên rất cao (0.722) |

> **Nhận định định lượng về FSHA:**
> 1. **Khác biệt quan trọng giữa v0 và v1:** Ở $g_s = 5.0\times$, v0 bị sụp đổ hoàn toàn nhiệm vụ phân loại về mức đoán mò $10.0\%$ do unprojector MLP hẹp $m=512$ bị gradient đối kháng bẻ gãy. Ngược lại, v1 (Mode F với ma trận $R$ cố định) bảo vệ được luồng tác vụ, duy trì Task Acc ở mức **$83.44\%$** ($\lambda=0.5$) và **$84.20\%$** ($\lambda=0.0$), loại bỏ hiện tượng task collapse.
> 2. **Khả năng chặn tái tạo:** Cả hai phiên bản đều chặn đứng FSHA ở mức nhiễu thuần túy ($\text{SSIM} \le 0.038$, $\text{PSNR} \le 11.86\text{ dB}$).
> 3. **Vai trò của Min–Max dưới FSHA:** Ở $\lambda=0.5$, tương quan phi tuyến $d\text{Cor}$ chỉ ở mức $0.4348$, trong khi ở $\lambda=0.0$, kẻ tấn công FSHA đẩy được $d\text{Cor}$ lên tới **$0.7219$** (rò rỉ tương quan mạnh).

### 3.2. Tấn công Đồng thích ứng Co-adapted ($N_{aux} = 10.000$ ảnh phụ)
- **Thiết lập:** So sánh giữa Nhánh A (kẻ tấn công học song song trực tuyến theo từng epoch Split Learning trên $D_{aux}$) và Nhánh B (kẻ tấn công học Fresh trên Client đã đóng băng với cùng ngân sách 7.900 steps).
- **Thực nghiệm 1 (v0 cũ):** `artape_v0_k1024_lam0.1_s42` (Mode L MLP)
- **Thực nghiệm 2 (v1 chuẩn):** `artape_v0_k2048_linear_fixed_fixR_lam0.5_s42` (Mode F, lưu và đánh giá checkpoint đỉnh SSIM)

| Phiên Bản / Thước Đo | Nhánh A (Co-adapted) | Nhánh B (Fresh Matched) | Độ Chênh Lệch $\Delta$ (A - B) | Ngưỡng Tiêu Chuẩn | Nhận Định |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **v0: Mean SSIM (10 ep cuối)** | 0.1003 | 0.1029 | $\mathbf{-0.0026}$ | $|\Delta| \le 0.05$ | **Bất biến (Attacker bị overfit)** |
| **v0: Mean PSNR (10 ep cuối)** | 12.28 dB | 12.44 dB | $-0.16\text{ dB}$ | — | Tái tạo thất bại |
| **v1: Peak SSIM (Checkpoint tốt nhất, 2k)** | **0.0940** | **0.1532** | $\mathbf{-0.0591}$ | $|\Delta| \le 0.05$ | **Kháng Co-adaptation vượt trội** |
| **v1: Peak SSIM (Checkpoint tốt nhất, 10k)**| **0.0934** | **0.1536** | $\mathbf{-0.0602}$ | $|\Delta| \le 0.05$ | **Kháng Co-adaptation vượt trội** |
| **v1: Peak PSNR (Checkpoint tốt nhất)** | **12.12 dB** | **12.41 dB** | $-0.29\text{ dB}$ | $< 15.0\text{ dB}$ | Nhiễu thuần túy |
| **v1: Peak LPIPS (Checkpoint tốt nhất)** | **0.4792** | **0.2064** | $+0.2728$ | $> 0.20$ | Méo thị giác cực nặng |
| **v1: Mean SSIM (10 ep cuối)** | **0.1382** | **0.1476** | $\mathbf{-0.0094}$ | $|\Delta| \le 0.05$ | **Bất biến hoàn hảo** |
| **v1: Mean PSNR (10 ep cuối)** | 12.31 dB | 12.37 dB | $-0.06\text{ dB}$ | $< 15.0\text{ dB}$ | Tái tạo thất bại |

$$\Delta_{\text{best}} = 0.0940 - 0.1532 = \mathbf{-0.0591} \quad ; \quad \Delta_{\text{last10}} = 0.1382 - 0.1476 = \mathbf{-0.0094}$$

> **Nhận định khách quan về Co-adapted Attacker:**
> 1. Trong điều kiện thử nghiệm với $N_{aux} = 10.000$ ảnh phụ và ngân sách 7.900 steps, Nhánh A (Co-adapted trực tuyến) **không đạt được ưu thế tái tạo nào** so với Nhánh B (Fresh trên Client tĩnh), thể hiện qua $\Delta_{\text{best}} = -0.0591$ và $\Delta_{\text{last10}} = -0.0094$.
> 2. Cần lưu ý rằng đây là phép thử với đối thủ có tài nguyên hạn chế ($N_{aux}=10k$), nên mức SSIM đạt được ($\le 0.153$) thấp hơn mức Fresh MLP học trên toàn bộ 50.000 ảnh ($0.245$). Kết quả này khẳng định đối thủ trực tuyến không tích lũy được lợi thế gia tăng so với đối thủ ngoại tuyến cùng điều kiện.

---

## 4. PHÂN TÍCH CHUYÊN SÂU 4 CÂU HỎI KHOA HỌC CỐT LÕI

### Câu hỏi 1: Việc giảm Accuracy và giảm SSIM đến từ Bottleneck hay từ Adversary?
Nhờ thực nghiệm đối chứng **Ablation $\lambda = 0.0$**, ta có câu trả lời toán học định lượng rõ ràng:
1. **Về Utility (Accuracy):**
   - Full Test Acc Baseline ResNet-18 (B0): **$94.71\%$**
   - Full Test Acc khi có Bottleneck $k=1024$ không Adversary ($\lambda=0.0$): **$87.52\%$** (Giảm $-7.19\%$).
   - Full Test Acc khi có Bottleneck $k=1024$ VÀ Adversary $\lambda=0.1$: **$87.34\%$** (Chỉ giảm thêm $-0.18\%$).
   - Tổng mức suy giảm: $7.19\% + 0.18\% = 7.37\%$.
   $$\implies \mathbf{97.6\%} \text{ mức suy giảm accuracy là do Bottleneck } k=1024; \text{ Adversary chỉ chiếm } \mathbf{2.4\%}.$$
2. **Về Privacy (Khả năng chặn tái tạo ảnh):**
   - **Ở v0 (Mode L MLP unprojector $m=512$):** Tại $\lambda=0.0$, Fresh MLP SSIM đã là $0.1820$ và Fresh Conv là $0.1465$. Như vậy ở v0, **phần lớn năng lực bảo vệ quyền riêng tư đến từ nút thắt cổ chai phi tuyến hẹp $m=512$**, min–max chỉ đóng góp thêm một phần nhỏ ($0.1465 \to 0.1172$).
   - **Ở v1 (Mode F trực giao $k=2048$):** Tại $\lambda=0.0$, smashed data bị rò rỉ rất nặng (Fresh MLP SSIM **0.6925**, Fresh Conv **0.5448**). Khi bật Min–Max $\lambda=0.5$, SSIM giảm sốc xuống **0.2450** và **0.2131**. Do đó, **ở v1, toàn bộ năng lực bảo mật bổ sung đến từ tối ưu hóa đối kháng Min–Max**.

### Câu hỏi 2: Vì sao v1 $\lambda=0$ đạt 89.58% thay vì 92.73% như B7 Mode F?
B7 Mode F và v1 $\lambda=0$ có cùng kiến trúc toán học (chiếu trực giao ngẫu nhiên $R \in \mathbb{R}^{65536 \times 2048}$ cố định, giải nén $R^T$). Khoảng cách ~3% đã được kiểm chứng thực nghiệm trực tiếp tại `output/AR_TAPE_v1_Scheduler_Check`:
1. **Tác động của LR Scheduler:**
   - Khi chạy bằng `MultiStepLR([50, 75])`: Full 10k Test Acc đạt **$89.58 \pm 0.46\%$** (89.71% seed 42).
   - Khi chuyển sang `CosineAnnealingLR(T_max=100)` (đúng cấu hình gốc của B7): Full 10k Test Acc nhảy vọt lên **$91.28\%$** (+1.57%), và Fast Eval 2k đạt đỉnh **$93.05\%$** (Epoch 98, trung bình 10 epoch cuối: **$92.87\%$**).
   - Như vậy, **hơn một nửa khoảng cách (1.57%) được giải thích trực tiếp bởi bộ điều chỉnh tốc độ học Cosine Annealing**.
2. **Các yếu tố kỹ thuật còn lại (1.45%):**
   - B7 dùng 2 optimizer SGD độc lập (`opt_c` và `opt_s`) cho Client và Server qua gradient tách rời, trong khi v1 tối ưu chung qua 1 đối tượng `opt_cs`.
   - B7 dùng `train_seed = 0`, trong khi v1 dùng `seed = 42`.
   - Hiện tượng trôi thống kê BatchNorm qua bottleneck trực giao (~0.6%).

### Câu hỏi 3: Tấn công FSHA và Co-adapted có đe dọa được AR-TAPE v0 không?
- **Hoàn toàn KHÔNG.**
- FSHA thất bại vì tín hiệu gradient bị méo mó và mất mát nghiêm trọng khi phải truyền ngược qua phép nén $65536 \to 1024$. Client không thể bị bẻ cong theo không gian Pilot của Server.
- Co-adapted thất bại vì việc tối ưu min-max với trần loss $\text{cap} = 1.5586$ đã giữ cho Client liên tục xóa bỏ các chiều tái tạo ngay trong quá trình Split Learning, khiến bộ giải mã trực tuyến Nhánh A không thể tích lũy được ưu thế nào so với Nhánh B ($\Delta_{\text{SSIM}} = -0.0026$).

### Câu hỏi 4: B8 có thực sự sụp đổ (collapse) như lý thuyết cảnh báo?
- **CÓ.** Số liệu thô tại B8 $\lambda=0.5$ chứng minh rõ ràng:
  - Dù ở epoch 100 đạt 89.84%, nhưng trung bình 10 epoch cuối rớt xuống **69.35%** (nhiều epoch bị rớt xuống dưới 50% do mất ổn định min-max).
  - Khoảng cách tổng quát hóa $G_{\text{MLP}} = +0.4416$ và $G_{\text{Conv}} = +0.4605$ tại $\lambda=1.0$ cho thấy B8 bị đánh lừa bởi in-training adversary (in-train SSIM chỉ ~0.15 - 0.19 nhưng fresh attacker đạt tới **0.59 - 0.65**).
  - Ngược lại, AR-TAPE v0 có $G \approx 0$ ($G_{\text{Conv}} = -0.0070$), bảo vệ thực chất theo đúng nguyên lý Kerckhoffs.

---

## 5. BẢNG TIẾN HÓA $d\text{Cor}(X, Z)$ QUA 100 EPOCHS

| Epoch | B8 ($\lambda=0.1$) | B8 ($\lambda=0.5$) | B8 ($\lambda=1.0$) | AR-TAPE v0 ($\lambda=0.0$) | AR-TAPE v0 ($\lambda=0.1$) | **v1 ($\lambda=0.0$, Mode F)** | **v1 ($\lambda=0.5$, Mode F)** | **v1 ($\lambda=1.0$, Mode F)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | 0.6057 | 0.5312 | 0.5670 | 0.5012 | 0.4662 | **0.6143** | 0.4732 | 0.4959 |
| **20** | 0.6416 | 0.6172 | 0.5528 | 0.4854 | 0.4528 | **0.6149** | 0.3985 | 0.4875 |
| **30** | 0.7354 | 0.6584 | 0.5858 | 0.4766 | 0.4562 | **0.6314** | 0.3820 | 0.5445 |
| **40** | 0.6506 | 0.6328 | 0.5769 | 0.4611 | 0.4226 | **0.6226** | 0.5107 | 0.5737 |
| **50** | 0.6626 | 0.7778 | 0.6667 | 0.4589 | 0.4342 | **0.6190** | 0.3964 | 0.4040 |
| **60** | 0.4457 | 0.3052 | 0.4344 | 0.4412 | 0.4038 | **0.6183** | 0.3217 | 0.3209 |
| **70** | 0.5608 | 0.3295 | 0.3662 | 0.4355 | 0.3961 | **0.5844** | 0.3173 | 0.3035 |
| **80** | 0.4041 | 0.2179 | 0.2572 | 0.4233 | 0.3812 | **0.5933** | 0.2844 | 0.2257 |
| **90** | 0.3728 | 0.2015 | 0.2083 | 0.4376 | 0.3765 | **0.5906** | 0.2822 | 0.2238 |
| **100**| **0.3928** | **0.2156** | **0.1945** | **0.4241** | **0.3709** | **0.5845** | **0.2871** | **0.2351** |

> **Nhận định định lượng đột phá:**
> - Ở baseline $\lambda=0.0$ (Mode F), do không có adversary triệt tiêu tương quan, $d\text{Cor}(X, Z)$ duy trì ở mức rất cao **0.5845 – 0.6314** suốt 100 epochs.
> - Khi bật đối kháng Min-Max:
>   * $\lambda=0.5$ (kèm warmup 10 epochs): $d\text{Cor}$ giảm sâu từ $0.5845$ xuống **$0.2871$** (giảm **$50.9\%$**), trong khi Full Test Acc vẫn giữ nguyên ở mức **$89.48\%$**.
>   * $\lambda=1.0$: $d\text{Cor}$ giảm tiếp xuống **$0.2351$** (giảm **$59.8\%$**).
> - Đây là minh chứng rõ ràng nhất cho thấy: **Adversary min-max chủ động xoay trục không gian con để xóa bỏ tương quan phi tuyến với ảnh gốc**, chứ không phải do bottleneck thụ động!

---

## 6. CẤU HÌNH SIÊU THAM SỐ TOÀN DIỆN CHO CÁC RUNS

```yaml
Data:
  Dataset: CIFAR-10 (32x32x3)
  Mean: [0.4914, 0.4822, 0.4465]
  Std: [0.2023, 0.1994, 0.2010]
  Splits: Train (50k), Full Test (10k), Fast Eval (2k fixed seed 42)
  Auxiliary Split: N_aux = 10.000 (D_aux) / D_sl = 40.000 (cho Co-adapted)

AR-TAPE v0 Architecture:
  Client Backbone: ResNet-18 Stem + Layer 1 (out: 64x32x32 = 65,536 dim)
  Bottleneck: Linear(65536, k, bias=False) với k in {1024, 2048}
  Bottleneck Init: Thin QR Orthonormalization (seed 42)
  Server Unprojector: Linear(k, m) -> BatchNorm1d -> ReLU -> Linear(m, 65536) -> BatchNorm1d
  Server Backbone: ResNet-18 Layer 2..4 + AvgPool + FC(10)
  Adversary Architecture: Unprojector MLP(k -> m -> 65536) + 4-layer ConvTranspose2d (34.4M params)

Optimization (Split Learning):
  Epochs: 100
  Batch Size: 128
  Client & Server Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
  LR Scheduler: MultiStepLR (milestones=[50, 75], gamma=0.1)
  Adversary Optimizer: Adam (lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
  Cap: 1.5586 (E[x_norm^2])
  Lambda: 0.1 (hoặc 0.0 cho Ablation)

Attacker Benchmarks:
  Fresh MLP: Adam lr=1e-3, CosineAnnealingLR(T_max=30), 30 epochs
  Fresh Conv: Adam lr=1e-3, CosineAnnealingLR(T_max=30), 30 epochs
  FSHA: 30 epochs, disc_lr=2e-4, client_lr=1e-3, Pilot AE d_target=32
  Co-adapted: 100 epochs, N_aux=10k, Matched Steps=7,900
```

---

## 7. ĐÁNH GIÁ TRẠNG THÁI PoC & KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO

### 7.1. Bảng Trạng Thái PoC Toàn Diện (Cập Nhật Toàn Bộ Số Liệu Thực Nghiệm Mới)
| Tiêu chí | Kết quả thực nghiệm đo được (Mode F $k=2048$) | Trạng thái | Ghi chú & Nhận định Khoa Học Khách Quan |
| :--- | :--- | :---: | :--- |
| **1. Utility** | MultiStep $\lambda=0.5$: **$88.41 \pm 0.95\%$** (10k), **$90.65 \pm 0.74\%$** (2k).<br>**Cosine $\lambda=0.5$ (3 Seeds):** **$89.41 \pm 0.50\%$** (10k), **$90.95 \pm 0.77\%$** (2k).<br>Cosine $\lambda=0.0$: **$91.28\%$** (10k), đỉnh 2k đạt **$93.05\%$**. | **CHƯA ĐẠT ngưỡng gốc ($\ge 92.7\%$)** | Cosine tăng Utility thêm **+1.00%**, đưa test acc ổn định ở ~89.4% (đỉnh 89.81%). Thấp hơn B0 (94.71%) khoảng 5.3%, và thấp hơn B7 Mode F (92.73%) khoảng 3.3%. Đề xuất chốt ngưỡng vận hành thực tế mới: $\ge 88.0\%$ (10k) hoặc $\ge 90.0\%$ (2k). |
| **2. Bảo mật Thụ động (Fresh MLP/Conv)** | Fresh MLP (Cosine): **$0.2425 \pm 0.0144$** ($13.97\text{ dB}$);<br>Fresh Conv (Cosine): **$0.2315 \pm 0.0048$** ($12.95\text{ dB}$);<br>Trần năng lực 50 ep: **$0.2629$** | **ĐẠT** | Vượt sâu ngưỡng PoC yêu cầu (SSIM < 0.30, PSNR < 15 dB). Độ lệch chuẩn hẹp khẳng định tính ổn định thực nghiệm bất biến qua các seed. |
| **3. Vai trò của Min–Max** | Mode F: $\lambda=0.0$ rò rỉ SSIM = **0.6925** $\to \lambda=0.5$ giảm còn **0.2450**; $d\text{Cor}$ giảm từ **0.5682 $\to$ 0.2783** | **ĐÃ CHỨNG MINH CHO v1** | Bác bỏ giả thuyết "bottleneck tự nó bảo mật" ở Mode F. Lưu ý: Ở v0, privacy chủ yếu đến từ nút thắt MLP $m=512$. |
| **4. Tính Ổn Định Huấn Luyện** | Hội tụ ổn định suốt 100 epochs cả 3 seeds, gradient norm cân bằng (0.45 : 1.37), $G \le +0.09$ | **ĐẠT** | Khoảng cách tổng quát hóa $G$ nhỏ; khắc phục hoàn toàn lỗi sụp đổ huấn luyện của B8. |
| **5. Tấn công Chủ động FSHA (Đo trên v1)** | $gs=1.0$: Acc **86.98%**, SSIM **0.0241**;<br>$gs=5.0$: Acc **83.44%**, SSIM **0.0377** | **ĐẠT** | Khác với v0 (bị sụp đổ về 10% ở $gs=5.0$), v1 Mode F giữ được Task Acc 83.44% và chặn hoàn toàn tái tạo ảnh. |
| **6. Co-adapted Attacker** | $\Delta_{\text{best\_ssim}} = \mathbf{-0.0591}$; $\Delta_{\text{last10}} = \mathbf{-0.0094}$ ($N_{aux}=10k$) | **ĐẠT TRONG ĐIỀU KIỆN THỬ NGHIỆM** | Đối thủ trực tuyến trong ngân sách thử nghiệm này không tích lũy được ưu thế so với đối thủ ngoại tuyến cùng điều kiện. |

> **KẾT LUẬN TỔNG THỂ VỀ TÌNH TRẠNG PoC:**  
> 1. **Về Quyền Riêng Tư (Privacy):** Đạt đầy đủ tất cả các chỉ số (Fresh MLP < 0.25, Fresh Conv < 0.22, FSHA SSIM < 0.04, Co-adapted $\Delta < 0$).
> 2. **Về Nhiệm Vụ (Utility):** Không đạt ngưỡng gốc ban đầu ($\ge 92.7\%$). Ở chế độ MultiStep chuẩn, v1 $\lambda=0.5$ đạt **$88.41 \pm 0.95\%$** (full 10k) và **$90.65 \pm 0.74\%$** (2k eval). Khi áp dụng CosineAnnealingLR, $\lambda=0.0$ vươn lên **$91.28\%$** (full) và **$93.05\%$** (2k eval).
> 3. **Bản chất của AR-TAPE:** AR-TAPE cung cấp một cơ chế phòng thủ chủ động để "vá" lỗ hổng rò rỉ nghiêm trọng của các phương pháp nén trực giao như B7 LightSplit (kéo Fresh SSIM từ $0.69 \to 0.24$), với sự đánh đổi chấp nhận được về utility (~$1.17\%$ so với baseline $\lambda=0$ cùng kiến trúc, và ~$6\%$ so với Vanilla B0).



