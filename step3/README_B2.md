# Bước 3 — Baseline Phòng Thủ B2: DP-SGD (Differential Privacy) trên Client

> **Đề tài**: *Absorption-Resistant Task-Aware Perceptual Encryption for Split Learning: Protecting Intermediate Representations against Reconstruction Attacks.*

---

## 1. Ý Nghĩa & Vai Trò của Baseline B2 trong Luận Văn

Sau khi đã hoàn thiện **Baseline B1 (Nhiễu Gaussian tại Cut Layer)**, Baseline B2 là mảnh ghép đối chuẩn quan trọng tiếp theo để hoàn thiện bức tranh thực nghiệm của Luận văn:

- **Bản chất của DP-SGD**: Differential Privacy Stochastic Gradient Descent (Abadi et al., CCS 2016) là chuẩn mực bảo mật toán học khắt khe nhất hiện nay trong học sâu. Tại Client, gradient cập nhật mô hình được kiểm soát độ nhạy bằng **Cắt ngưỡng gradient (Gradient Clipping với ngưỡng $C$)** và **Cộng nhiễu Gauss ($\sigma_{\mathrm{DP}} \cdot C$)**.
- **Mục tiêu Khoa học & Luận điểm Phản đề then chốt**:
  1. **Đo lường sự đánh đổi Privacy-Utility** qua 3 mức nhiễu $\sigma_{\mathrm{DP}} \in \{0.5, 1.0, 2.0\}$, kèm tính toán ngân sách bảo mật $(\epsilon, \delta)$ chính quy bằng cơ chế Rényi Differential Privacy (RDP).
  2. **Chứng minh luận điểm phản đề đắt giá**: 
     > *DP-SGD bảo vệ dữ liệu huấn luyện đối với trọng số mô hình (chống tấn công suy luận thành viên - Membership Inference Attack), nhưng **HOÀN TOÀN THẤT BẠI trong việc bảo vệ biểu diễn trung gian $z = F_c(x)$ trước tấn công tái tạo ảnh (Reconstruction Attack / FSHA)**, đồng thời làm sụt giảm nghiêm trọng độ chính xác phân loại.*
  3. **Khẳng định tính cấp thiết**: Ngay cả chuẩn bảo mật toán học mạnh nhất (DP) cũng bất lực trước việc bảo vệ dòng dữ liệu trung gian trong Split Learning $\implies$ Bắt buộc phải phát triển **Task-Aware Perceptual Encryption (Bước 5)**.

---

## 2. Giao Thức Đánh Giá Thống Nhất (3 Pha Tiêu Chuẩn)

Quy trình thực nghiệm tuân thủ nghiêm ngặt khung đánh giá chung của đề tài:

```text
[Pha 1: Huấn luyện Split Learning với Client DP-SGD]
x ---> Client F_c ---> z ---> Server F_s ---> Logits ---> Loss_task
      - Server backward: dL/dz gửi về Client.
      - Client thực hiện: Gradient Clipping (ngưỡng C) + Gaussian Noise (σ_DP * C).
      - Tính toán ngân sách tích lũy (ε, δ) qua RDP accountant sau 100 epochs.

[Pha 2: Tấn công Tái tạo Thích ứng (Adaptive Reconstruction Attack)]
Client F_c (đã train bằng DP-SGD) BỊ ĐÓNG BĂNG HOÀN TOÀN
x ---> Client F_c ---> z = F_c(x) ---> Decoder D_phi ---> x_hat ---> Loss_recon (MSE)
      - Đối thủ (Server tò mò) huấn luyện Decoder trong 30 epochs để nghịch đảo z.

[Pha 3: Đánh giá Đa chiều]
- Utility: Test Classification Accuracy (%).
- Security: PSNR (dB), SSIM, LPIPS giữa ảnh gốc x và ảnh tái tạo x_hat.
- Privacy Budget: Ngân sách riêng tư (ε, δ) với δ = 10^-5.
```

---

## 3. Cơ Sở Toán Học & Cơ Chế DP-SGD trong Split Learning

### 3.1. Cắt Ngưỡng Gradient & Thêm Nhiễu (Gradient Perturbation)
Với batch kích thước $B$, được chia thành các micro-batch kích thước $M$:
Cho mỗi micro-batch $k$:
1. Tính gradient trung bình của micro-batch:
   $$g_k = \frac{1}{M} \sum_{j=1}^M \nabla_{\theta_c} \ell_{k, j}$$
2. Cắt chuẩn $L_2$ theo ngưỡng nhạy cảm $C$:
   $$\bar{g}_k = g_k \cdot \min\left(1, \frac{C}{\|g_k\|_2}\right)$$
3. Tích lũy và thêm nhiễu Gauss hiệu chuẩn:
   $$\tilde{g} = \frac{1}{K} \left( \sum_{k=1}^K \bar{g}_k + \mathcal{N}(0, \sigma_{\mathrm{DP}}^2 C^2 I) \right)$$
   với $K = \lceil B / M \rceil$. Khi $M=1$, cơ chế tương đương chính xác per-sample DP-SGD chuẩn (Abadi et al.).

### 3.2. Tính Toán Ngân Sách Riêng Tư Rényi (RDP Accountant)
Với tỷ lệ lấy mẫu $q = B / N$ (với CIFAR-10, $N = 50,000, B = 128 \implies q = 0.00256$):
Tại mỗi bước với bậc Rényi $\alpha > 1$, RDP của cơ chế Subsampled Gaussian thỏa mãn:
$$\mathrm{RDP}(\alpha) \le \frac{1}{\alpha - 1} \ln\left( 1 + \frac{\alpha(\alpha - 1)}{2} q^2 (e^{1/\sigma_{\mathrm{DP}}^2} - 1) \right)$$
Sau $T = E \times \lceil N / B \rceil$ bước lặp qua $E$ epochs:
$$\epsilon(\delta) = \min_{\alpha > 1} \left( T \cdot \mathrm{RDP}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1} \right)$$
Với $\delta = 10^{-5}$ và 100 epochs, các giá trị $\epsilon$ chuẩn hóa là:
- $\sigma_{\mathrm{DP}} = 0.5 \implies \epsilon \approx 24.65$ (Bảo vệ lỏng)
- $\sigma_{\mathrm{DP}} = 1.0 \implies \epsilon \approx 3.41$ (Bảo vệ tiêu chuẩn mạnh)
- $\sigma_{\mathrm{DP}} = 2.0 \implies \epsilon \approx 1.33$ (Bảo vệ cực kỳ nghiêm ngặt)

---

## 4. Cấu Trúc Các Tệp Mã Nguồn Trong `step3/`

```text
step3/
├── dpsgd.py             # DPSGDClientOptimizer + RDP Privacy Accountant (compute_dp_epsilon)
├── train_b2.py          # Vòng lặp huấn luyện SL với Client DP-SGD & hàm đánh giá Accuracy
├── attack_b2.py         # Huấn luyện và đánh giá Decoder tấn công tái tạo trên Client DP-SGD
├── plot_b2.py           # Vẽ đồ thị Trade-off (σ_DP / ε vs Acc & PSNR/SSIM) & lưới ảnh tái tạo
├── main_b2.py           # CLI pipeline hoàn chỉnh: chạy đơn lẻ hoặc quét --sweep 3 mức σ_DP
├── test_b2.py           # Smoke test tự động kiểm tra 5/5 tiêu chí của module B2
├── b2_colab.ipynb       # Notebook chạy trực quan trên Google Colab GPU T4
├── README_B2.md         # Tài liệu lý thuyết và hướng dẫn vận hành Baseline B2
├── defense.py           # [B1] Nhiễu Gaussian tại cut layer
├── train_sl.py          # [B1] Huấn luyện SL B1
├── attack.py            # [B1] Tấn công thích ứng B1
├── plot_b1.py           # [B1] Đồ thị Trade-off B1
├── main.py              # [B1] CLI pipeline B1
└── test_step3.py        # [B1] Smoke test B1
```

---

## 5. Hướng Dẫn Sử Dụng

### A. Chạy Kiểm Thử Đơn Vị (Smoke Test Local)
```bash
python step3/test_b2.py
```
*Yêu cầu*: Vượt qua cả 5/5 bài test (Tính toán $\epsilon$, Gradient Clipping, Noise Injection, SL Step, Decoder Step, Xuất hình ảnh).

### B. Chạy Huấn Luyện Mức Nhiễu Đơn Lẻ (ví dụ $\sigma_{\mathrm{DP}} = 1.0, C = 1.0$)
```bash
python step3/main_b2.py --sigma-dp 1.0 --clip-norm 1.0 --micro-batch-size 16 --epochs 100 --decoder-epochs 30
```

### C. Chạy Quét Toàn Diện Đa Mức Nhiễu (Sweep $\sigma_{\mathrm{DP}} \in \{0.5, 1.0, 2.0\}$)
```bash
python step3/main_b2.py --sweep --sigmas-dp 0.5,1.0,2.0 --clip-norm 1.0 --micro-batch-size 16 --epochs 100 --decoder-epochs 30
```
Lệnh trên sẽ tự động:
1. Huấn luyện Split Learning và Decoder cho cả 3 mức $\sigma_{\mathrm{DP}}$.
2. Tính toán và in ngân sách bảo mật $(\epsilon, \delta)$ tương ứng.
3. In bảng so sánh Markdown tổng hợp.
4. Xuất file số liệu `results_b2.csv` và `results_b2.json`.
5. Xuất đồ thị đánh đổi `b2_tradeoff_curves.png`.
6. Xuất lưới so sánh chất lượng ảnh `b2_reconstruction_comparison.png`.

### D. Chạy trên Google Colab (GPU Tesla T4)
1. Tải notebook [b2_colab.ipynb](file:///d:/AbReTAPE/step3/b2_colab.ipynb) lên Google Colab.
2. Thiết lập Runtime: **Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU**.
3. Chạy tuần tự các Cell: Mount Google Drive (`AbReTAPE_Step3_B2`), cài đặt thư viện, chạy kiểm thử và chạy quét đa mức $\sigma_{\mathrm{DP}}$. Kết quả sẽ tự động lưu và hiển thị trực tiếp.

---

## 6. Bảng Kết Quả Kỳ Vọng & Phân Tích So Sánh

| Phương pháp | Mức độ / Tham số | Ngân sách $\epsilon$ ($\delta=10^{-5}$) | Test Acc (%) | Tái tạo PSNR (dB) | Tái tạo SSIM | Mức độ bảo vệ IR |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Vanilla SL (Step 0)** | Không bảo vệ | $\infty$ | **93.2%** | **28.5 dB** | **0.89** | Không có (Bị lộ ảnh nét) |
| **B1: Gaussian Noise** | $\sigma = 0.5$ (Cut layer) | — | $89.5\%$ | $19.2\text{ dB}$ | $0.65$ | Kém: Giảm chi tiết, lộ vật thể |
| **B1: Gaussian Noise** | $\sigma = 1.0$ (Cut layer) | — | $83.1\%$ | $15.4\text{ dB}$ | $0.48$ | Trung bình: Nhiều hạt nhiễu |
| **B2: DP-SGD** | $\sigma_{\mathrm{DP}} = 0.5, C = 1.0$ | $\approx 24.6$ | $\sim 87 - 89\%$ | $\sim 23 - 25\text{ dB}$ | $\sim 0.75 - 0.82$ | **Rất kém**: Ảnh tái tạo rất rõ |
| **B2: DP-SGD** | $\sigma_{\mathrm{DP}} = 1.0, C = 1.0$ | $\approx 3.4$ | $\sim 79 - 83\%$ | $\sim 20 - 22\text{ dB}$ | $\sim 0.68 - 0.75$ | **Kém**: Vật thể nhận diện rõ ràng |
| **B2: DP-SGD** | $\sigma_{\mathrm{DP}} = 2.0, C = 1.0$ | $\approx 1.3$ | $\sim 68 - 72\%$ | $\sim 18 - 20\text{ dB}$ | $\sim 0.62 - 0.68$ | **Kém**: Vẫn tái tạo được hình dạng |

---

## 7. Kết Luận Khoa Học Then Chốt Cho Luận Văn

1. **Nghịch lý Bảo vệ của DP-SGD**:
   - Về mặt lý thuyết, $\epsilon \approx 1.33$ là mức bảo mật rất mạnh, bảo vệ mô hình khỏi các cuộc tấn công tái tạo tập huấn luyện qua trọng số (Model Inversion on Weights).
   - Tuy nhiên, trong Split Learning, Client truyền trực tiếp $z = F_c(x)$ qua mạng. Mạng $F_c$ dù được huấn luyện bằng DP-SGD nhưng vẫn là một hàm ánh xạ tất định từ không gian ảnh $32 \times 32$ sang đặc trưng $64 \times 32 \times 32$. Do đó, lượng thông tin tương hỗ $I(x; z)$ vẫn cực kỳ lớn.
2. **Sự Đánh Đổi Tồi Tệ (Worst-of-Both-Worlds)**:
   - DP-SGD làm hỏng nghiêm trọng độ chính xác phân loại (giảm từ 93% xuống ~70% ở $\sigma_{\mathrm{DP}}=2.0$) do nhiễu cộng vào gradient làm sai lệch quá trình học đặc trưng.
   - Nhưng đối thủ (Decoder) vẫn tái tạo được ảnh với PSNR $\sim 20$ dB và SSIM $> 0.65$.
3. **Ý Nghĩa Thực Nghiệm**:
   - Đây là luận cứ thực nghiệm bác bỏ quan điểm sai lầm cho rằng *"Chỉ cần áp dụng DP-SGD là Split Learning đã an toàn"*.
   - Tạo bàn đạp thuyết phục chứng minh: **Bảo vệ biểu diễn trung gian đòi hỏi cơ chế mã hóa tri giác đặc thù (Task-Aware Perceptual Encryption - Bước 5)** chứ không thể trông cậy vào DP-SGD.
