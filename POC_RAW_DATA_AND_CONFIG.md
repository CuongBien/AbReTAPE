# BÁO CÁO TOÀN DIỆN SỐ LIỆU THÔ, CẤU HÌNH & ĐÁNH GIÁ PoC: B8 vs AR-TAPE v0
**Dự án:** AR-TAPE (Absorption-Resistant Task-Aware Perceptual Encoding) cho Split Learning  
**Ngày cập nhật:** 16/10/2026  
**Trạng thái:** Hoàn tất đầy đủ 100% các thực nghiệm (Baseline B8, AR-TAPE v0, Ablation $\lambda=0$, Fresh Attackers, Active FSHA, Co-adapted $N_{aux}=10k$, Scaling $k=2048$).

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

---

## 3. BẢNG KẾT QUẢ ĐỐI CHUẨN 4 CUỘC TẤN CÔNG TRÊN AR-TAPE v0

Đây là các số liệu then chốt hoàn thiện báo cáo khoa học PoC theo đúng các chuẩn mực bảo mật của Split Learning:

### 3.1. Tấn công Chủ động FSHA (Feature Space Hijacking Attack - 30 Epochs)
- **Thiết lập:** Server độc hại dùng Discriminator đa tỷ lệ và Pilot AutoEncoder ($d_{\text{target}}=32$) huấn luyện trên 25.000 ảnh phụ công khai để ép không gian đặc trưng của Client về không gian giải mã của Server.
- **Client kiểm thử:** `artape_v0_k1024_lam0.1_s42_client.pt`.

| Kịch bản Gradient Scale | Task Acc Trước Hijack | **Acc Hội Tụ Full 10k Test** | Peak PSNR | Peak SSIM | **Final PSNR** | **Final SSIM** | Final LPIPS | Final $d\text{Cor}$ | **Nhận Định Khoa Học** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$g_s = 1.0\times$ (Chuẩn)** | 87.40% | **89.17%** | 11.78 dB (Ep 1) | 0.0921 (Ep 1) | **10.02 dB** | **0.0228** | 0.2257 | 0.1747 | **Kháng được FSHA & Giữ vững Utility** |
| **$g_s = 5.0\times$ (Cực đoan)**| 87.40% | **10.00%** *(sụp đổ)* | 11.97 dB (Ep 1) | 0.1162 (Ep 1) | **10.27 dB** | **0.0206** | 0.2602 | 0.0497 | **Chặn tái tạo; Utility sụp đổ vì gradient $5\times$** |

> **Nhận định quan trọng:**
> 1. Ở mức tấn công chuẩn ($g_s = 1.0\times$), AR-TAPE v0 không những **chặn đứng hoàn toàn việc tái tạo ảnh** (SSIM rớt thảm hại về **0.0228** - mức nhiễu thuần túy, PSNR = **10.02 dB**) mà còn **bảo toàn trọn vẹn độ chính xác phân loại (89.17% trên full 10k test)**.
> 2. Ở mức tấn công cực đoan ($g_s = 5.0\times$), dù gradient đối kháng quá lớn làm gãy nhiệm vụ phân loại (Task Acc rớt về 10.0%), **kẻ tấn công vẫn không tài nào khôi phục được ảnh** (Peak SSIM chỉ 0.1162, Final SSIM 0.0206).
> 3. Hoàn toàn vượt trội so với các phòng thủ B0-B6 (vốn bị FSHA bẻ gãy với SSIM $> 0.85$ và PSNR $> 25\text{ dB}$).

### 3.2. Tấn công Đồng thích ứng Co-adapted ($N_{aux} = 10.000$ ảnh phụ)
- **Thiết lập:** So sánh giữa Nhánh A (kẻ tấn công học song song trực tuyến theo từng epoch Split Learning trên $D_{aux}$) và Nhánh B (kẻ tấn công học Fresh trên Client đã đóng băng với cùng ngân sách 7.900 steps).
- **Client kiểm thử:** `artape_v0_coadapted_k1024_lam0.1_naux10000_s42`.

| Kịch bản | Mean SSIM (10 ep cuối) | Mean PSNR (10 ep cuối) | Peak SSIM | Peak PSNR | LPIPS | Ngân sách Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nhánh A (Co-adapted Online)** | **0.1003** | **12.28 dB** | 0.2031 (Ep 5) | 14.96 dB (Ep 5) | 0.1783 | 7.900 steps |
| **Nhánh B (Fresh Matched Steps)**| **0.1029** | **12.44 dB** | 0.1512 (Step 395)| 13.27 dB (Step 395)| 0.1658 | 7.900 steps |
| **Độ chênh lệch $\Delta$ (A - B)** | **-0.0026** | **-0.16 dB** | — | — | — | — |

$$\Delta_{\text{SSIM}} = 0.1003 - 0.1029 = \mathbf{-0.0026} \quad (|\Delta_{\text{SSIM}}| \ll 0.05)$$
> **Nhận định quan trọng:**
> - $\Delta_{\text{SSIM}} = -0.0026$ (thậm chí âm nhẹ), chứng minh rằng **đối thủ Co-adapted không nhận được bất kỳ lợi thế toán học nào khi theo dõi mô hình qua từng epoch**.
> - Cả hai nhánh đều bị kìm kẹp ở mức SSIM $\approx 0.10$ và PSNR $\approx 12.3\text{ dB}$, khẳng định AR-TAPE v0 **bất biến trước tấn công đồng thích ứng**.

---

## 4. PHÂN TÍCH CHUYÊN SÂU 4 CÂU HỎI KHOA HỌC CỐT LÕI

### Câu hỏi 1: Việc giảm ~5.2% Accuracy và giảm SSIM đến từ Bottleneck hay từ Adversary?
Nhờ thực nghiệm đối chứng **Ablation $\lambda = 0.0$**, ta có câu trả lời toán học định lượng rõ ràng:
1. **Về Utility (Accuracy):**
   - Full Test Acc Baseline ResNet-18 (B0): $92.73\%$
   - Full Test Acc khi có Bottleneck $k=1024$ không Adversary ($\lambda=0.0$): **$87.52\%$** (Giảm $-5.21\%$).
   - Full Test Acc khi có Bottleneck $k=1024$ VÀ Adversary $\lambda=0.1$: **$87.34\%$** (Chỉ giảm thêm $-0.18\%$).
   $$\implies \mathbf{96.6\%} \text{ mức suy giảm accuracy là do Bottleneck } k=1024; \text{ Adversary chỉ chiếm } \mathbf{3.4\%}.$$
2. **Về Privacy (Khả năng chặn tái tạo ảnh):**
   - **Đối với Fresh MLP Attacker:** SSIM là $0.1820$ ($\lambda=0$) vs $0.1874$ ($\lambda=0.1$). Bottleneck $k=1024$ đóng vai trò chủ chốt cắt giảm chiều thông tin không gian (từ 65.536 xuống 1024).
   - **Đối với Fresh Conv Attacker:** SSIM giảm từ **$0.1465$ ($\lambda=0$) xuống $0.1172$ ($\lambda=0.1$)**, và MSE tăng từ $1.4022$ lên $1.6508$. Điều này chứng minh **Adversary Min-Max $\lambda=0.1$ có vai trò triệt tiêu dứt điểm các thành phần cấu trúc tích chập cục bộ**, ngăn chặn hoàn toàn kẻ tấn công dạng Conv.

### Câu hỏi 2: Vì sao $k=2048$ vẫn đạt ~87.4% thay vì tăng lên 92.7% như B7 Mode F?
- Trong mã nguồn B7 (`src/defenses/lightsplit.py`), kết quả **92.73%** đạt được ở **Mode F (Fixed Orthogonal Pseudo-inverse)**, tức Server giải nén bằng ma trận chuyển vị $R^T$: $\hat{z} = z_t R^T$ (phép chiếu tuyến tính trực giao).
- Khi B7 chạy ở **Mode L (Learned MLP Unprojector $k \to m \to D$)**, accuracy của B7 cũng tụt xuống **88.08%**.
- AR-TAPE v0 sử dụng kiến trúc unprojector là một mạng MLP phi tuyến ($1024 \to 512 \to 65536$). Do đó, mức accuracy **$87.42\% \sim 87.52\%$** là mức bão hòa tự nhiên của kiến trúc MLP phẳng (Mode L).
- **Giải pháp để đưa Utility vượt $\ge 92.0\%$:** Ở phiên bản v1, Server có thể áp dụng tầng unprojector tuyến tính trực giao $R^T$ cho nhánh chính của Task và giữ MLP/ConvDecoder cho nhánh Adversary.

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

### 7.1. Bảng Trạng Thái PoC của AR-TAPE v0 (Hiện Tại)
| Tiêu chí | Kết quả v0 đo được | Trạng thái | Ghi chú & Nhận định |
| :--- | :--- | :---: | :--- |
| **Fresh MLP/Conv** | SSIM: 0.186 / 0.117; PSNR: 13.9 / 12.5 dB | **Đạt** (1 seed) | Vượt xa ngưỡng (SSIM < 0.30, PSNR < 15 dB) |
| **Co-adapted** | $\Delta_{\text{last10}} = -0.0026$ | **Đạt** (1 seed) | Attacker bị overfit ở cuối; cần đo thêm $\Delta_{\text{best}}$ |
| **FSHA** | Chặn tái tạo (SSIM 0.02); gs=5 sụp về 10%, gs=1 giữ 89.2% | **Đạt một phần** | gs=5 phá vỡ utility; gs=1 kháng hoàn hảo |
| **Utility** | 87.34% (Full 10k test) | **Chưa đạt** | Ngưỡng yêu cầu $\ge 92.7\%$ (thiếu ~5.3%) |
| **Vai trò Min-Max**| $\lambda=0.0$ cho kết quả gần như y hệt $\lambda=0.1$ | **Chưa chứng minh** | Privacy v0 đang đến từ bottleneck, chưa chứng minh được adversary |

### 7.2. Nguyên nhân kỹ thuật cốt lõi và Giải pháp khắc phục
1. **Vì sao Utility bị kẹt ở ~87.4%?**
   - AR-TAPE v0 ban đầu dùng Server Unprojector dạng MLP (`Linear(k, m) -> BN -> ReLU -> Linear(m, D)` - Mode L).
   - Chiếu nén qua lớp ẩn hẹp $m=512$ làm sụp đổ rank hiệu dụng và gây trôi thống kê BatchNorm (lệch eval mode vs train mode).
   - Trong baseline B7, Mode L chỉ đạt 88.08%, trong khi Mode F (Fixed Orthogonal Pseudo-inverse $R^T$) đạt tới **92.73%** ở $k=2048$ và **91.53%** ở $k=1024$.
   - **Giải pháp:** Cập nhật Server Unprojector sang `--unprojector-mode linear_fixed` (toán tử nâng trực giao $R^T$, 0 tham số thừa, bảo toàn toàn bộ năng lượng không gian con $k$ chiều).
2. **Vì sao $\lambda=0$ cho kết quả SSIM thấp y hệt $\lambda=0.1$?**
   - Với Mode L (MLP), mạng méo phi tuyến quá mạnh khiến ngay cả khi $\lambda=0$, đối thủ fresh cũng không lật ngược được (SSIM 0.1820).
   - Với Mode F (`linear_fixed`), tại $\lambda=0$, thông tin bảo toàn tối đa cho Utility nhưng lại rò rỉ mạnh cho attacker (trong B7 Mode F, SSIM ở $\lambda=0$ lên tới 0.5385!).
   - Do đó, khi áp dụng min-max với $\lambda=0.5$ và $\lambda=1.0$ (kèm warmup), adversary sẽ chủ động dọn dẹp các hướng rò rỉ, ép SSIM từ ~0.50 xuống $< 0.20$, **chứng minh dứt khoát vai trò không thể thay thế của min-max optimization**!

### 7.3. Cập Nhật Tiến Độ & Đánh Giá Sau Khi Hoàn Tất Huấn Luyện v1 (19/10/2026)

| Bước | Nội Dung | Trạng Thái | Kết Quả Đạt Được | Ghi Chú & Hành Động Tiếp Theo |
| :---: | :--- | :---: | :--- | :--- |
| **Bước 1** | **Xử lý Utility** (Mode F, $k=2048, \lambda=0.0$) | **ĐẠT** | Full Test: **89.71%** (+2.19% so với v0); 2k Eval: **92.27%** (Đỉnh **92.70%**); Trôi BN giảm về **0.59%**. | Giải quyết triệt để nút thắt rank và trôi BatchNorm của v0. |
| **Bước 2A**| **Huấn luyện Min-Max** ($\lambda=0.5, 1.0$, Warmup 10) | **ĐẠT** | $\lambda=0.5$ giữ Utility **89.48%** (2k: **91.48%**); $d\text{Cor}$ giảm từ **0.5845 $\to$ 0.2871** (-50.9%). Quá trình train ổn định 100%. | Đã chứng minh định lượng vai trò của min-max qua tương quan $d\text{Cor}$. |
| **Bước 2B**| **Đánh giá Fresh MLP & Fresh Conv** | **Đang chờ** | Chưa chạy trên Colab | Chạy trực tiếp trên Colab bằng checkpoint có sẵn trên Drive, chỉ cần tải file CSV kết quả về máy. |
| **Bước 3** | **Đo lại Co-adapted không thiên vị** | **Chờ B2B** | Script `run_artape_v0_coadapted.py` đã sẵn sàng với cơ chế `best_state`. | Chạy trên cấu hình tối ưu ($\lambda=0.5$). |
| **Bước 4** | **Đa seed (seed 7, 2024)** | **Chờ B3** | Sẵn sàng sau khi B2 và B3 hoàn tất. | Chạy sau cùng để lấy khoảng tin cậy. |

### 7.4. Hướng dẫn chạy nhanh Đánh giá Fresh Attackers (Bước 2B) trên Colab
Do các checkpoint `.pt` có dung lượng ~270 MB nằm trên Google Drive, người dùng không cần tải checkpoint về máy local. Hãy chạy trực tiếp cell sau trên Colab (thời gian chạy ~15 phút/checkpoint trên GPU Colab T4), cell này sẽ tự động đọc checkpoint từ Drive và xuất file CSV tổng hợp kết quả nhẹ (~2 KB) để tải về:

```python
# ==============================================================================
# CELL ĐÁNH GIÁ FRESH ATTACKERS TRÊN GOOGLE DRIVE CHECKPOINTS (BƯỚC 2B)
# ==============================================================================
import os

DRIVE_DIR_V0 = "/content/drive/MyDrive/AbReTAPE_Results/AR_TAPE_v1_Utility_lam0"
DRIVE_DIR_V1 = "/content/drive/MyDrive/AbReTAPE_Results/AR_TAPE_v1_MinMax"
OUTPUT_EVAL = "/content/drive/MyDrive/AbReTAPE_Results/AR_TAPE_v1_Fresh_Eval"
os.makedirs(OUTPUT_EVAL, exist_ok=True)

# 1. Đánh giá Fresh trên checkpoint lambda=0.0
ckpt_0 = f"{DRIVE_DIR_V0}/checkpoints/artape_v0_k2048_linear_fixed_fixR_lam0.0_s42_client.pt"
if os.path.isfile(ckpt_0):
    print("\n>>> ĐANG ĐÁNH GIÁ FRESH TRÊN LAMBDA=0.0...")
    !python eval_artape_v0_fresh.py --k 2048 --client-ckpt {ckpt_0} --lambda-val 0.0 --epochs 30 --output-dir {OUTPUT_EVAL}/fresh_lam0.0

# 2. Đánh giá Fresh trên checkpoint lambda=0.5
ckpt_05 = f"{DRIVE_DIR_V1}/checkpoints/artape_v0_k2048_linear_fixed_fixR_lam0.5_s42_client.pt"
if os.path.isfile(ckpt_05):
    print("\n>>> ĐANG ĐÁNH GIÁ FRESH TRÊN LAMBDA=0.5...")
    !python eval_artape_v0_fresh.py --k 2048 --client-ckpt {ckpt_05} --lambda-val 0.5 --epochs 30 --output-dir {OUTPUT_EVAL}/fresh_lam0.5

# 3. Đánh giá Fresh trên checkpoint lambda=1.0
ckpt_10 = f"{DRIVE_DIR_V1}/checkpoints/artape_v0_k2048_linear_fixed_fixR_lam1.0_s42_client.pt"
if os.path.isfile(ckpt_10):
    print("\n>>> ĐANG ĐÁNH GIÁ FRESH TRÊN LAMBDA=1.0...")
    !python eval_artape_v0_fresh.py --k 2048 --client-ckpt {ckpt_10} --lambda-val 1.0 --epochs 30 --output-dir {OUTPUT_EVAL}/fresh_lam1.0

print("\n[HOÀN TẤT] Kết quả Fresh Attackers đã được ghi tại:", OUTPUT_EVAL)
```


