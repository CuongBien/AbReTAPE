# Bước 3 — Baseline Phòng Thủ B1: Nhiễu Gaussian tại Cut Layer

> **Đề tài**: *Absorption-Resistant Task-Aware Perceptual Encryption for Split Learning: Protecting Intermediate Representations against Reconstruction Attacks.*

---

## 1. Ý Nghĩa & Vai Trò của Bước 3 trong Đề Tài
Bước 3 mở đầu chuỗi đánh giá các cơ chế phòng thủ kinh điển (**Baselines B1–B6**) để làm cơ sở đối chuẩn trực tiếp với **Encoder Task-Aware phi khả nghịch** được đề xuất trong Bước 5:
- **Baseline 1 (B1 - Gaussian Noise)**: Là giải pháp bảo vệ dữ liệu đơn giản, phổ biến và ít tốn kém tính toán nhất. Phía Client cộng thêm nhiễu Gauss độc lập $\mathcal{N}(0, \sigma^2 I)$ vào biểu diễn trung gian $z$ trước khi truyền cho Server.
- **Mục tiêu cốt lõi**:
  1. Xác lập **Khung Đánh Giá Thống Nhất (Unified Evaluation Protocol)** dùng chung cho toàn bộ các baseline B1–B6.
  2. Đo lường sự đánh đổi giữa **Chất lượng Tác vụ (Utility - Accuracy)** và **Tính An toàn (Security - PSNR, SSIM, LPIPS)** qua ít nhất 3 mức nhiễu $\sigma \in \{0.1, 0.5, 1.0\}$.
  3. Chỉ ra điểm yếu chí mạng của phòng thủ nhiễu: Giảm chất lượng ảnh tái tạo nhưng **hoàn toàn không ngăn chặn được đối thủ thích ứng (Adaptive Adversary)** — Decoder được huấn luyện trên IR có nhiễu vẫn khôi phục được phần lớn hình dạng và thông tin ngữ nghĩa. Đây là luận điểm quyết định chứng minh sự cần thiết của mã hóa cảm nhận Task-Aware.

---

## 2. Giao Thức Đánh Giá Thống Nhất (Unified Protocol cho B1–B6)
Mọi phương pháp phòng thủ từ B1 đến B6 đều tuân theo đúng 3 pha thực nghiệm tiêu chuẩn:

```
[Pha 1: SL Training]
x ---> Client F_c ---> z ---> [Phòng thủ B1: z + σ*ε] ---> z' ---> Server F_s ---> Logits ---> Loss_task
      (100 epochs, cập nhật gradient cả Client và Server qua biên giới phòng thủ)

[Pha 2: Adaptive Attack]
Client F_c & Phòng thủ B1 BỊ ĐÓNG BĂNG
x ---> Client F_c ---> z' (đã phòng thủ) ---> Decoder D_phi ---> x_hat ---> Loss_recon (MSE)
      (30 epochs, kẻ tấn công tối ưu hóa Decoder trên IR phòng thủ)

[Pha 3: Đánh giá Đa chiều]
- Utility: Test Classification Accuracy (%)
- Security: PSNR (dB), SSIM, LPIPS giữa x và x_hat
```

---

## 3. Cơ Sở Toán Học & Cơ Chế Lan Truyền Gradient

### 3.1. Phép Cộng Nhiễu Cut Layer
Biểu diễn trung gian sau phòng thủ:
$$z' = z + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
trong đó:
- $z = F_c(x) \in \mathbb{R}^{B \times 64 \times 32 \times 32}$ là IR gốc từ Client.
- $\sigma \in \{0.1, 0.5, 1.0\}$ là độ lệch chuẩn kiểm soát cường độ nhiễu.

### 3.2. Gradient qua Nhiễu về Client
Khi lan truyền ngược từ Server về Client:
$$\frac{\partial \mathcal{L}_{\text{task}}}{\partial z} = \frac{\partial \mathcal{L}_{\text{task}}}{\partial z'} \cdot \frac{\partial z'}{\partial z} = \frac{\partial \mathcal{L}_{\text{task}}}{\partial z'} \cdot I = \frac{\partial \mathcal{L}_{\text{task}}}{\partial z'}$$
Gradient được truyền thẳng qua phép cộng nhiễu mà không bị triệt tiêu, cho phép Client tự học cách biểu diễn các đặc trưng có khả năng chống chịu nhiễu Gauss.

### 3.3. Tấn công Tái tạo Thích ứng (Adaptive Decoder Attack)
Kẻ tấn công đóng vai trò Server tò mò nhưng trung thực, thu thập $z'$ và huấn luyện mạng Decoder $D_\phi$:
$$\min_{\phi} \; \mathbb{E}_{x \sim \mathcal{D}} \Big[ \big\| D_\phi\big(F_c(x) + \sigma \epsilon\big) - x \big\|_2^2 \Big]$$

---

## 4. Cấu Trúc Thư Mục `step3/`

```text
step3/
├── defense.py               # Module GaussianNoise(sigma) tại cut layer
├── train_sl.py              # Vòng lặp SL huấn luyện với nhiễu Gauss
├── attack.py                # Huấn luyện & đánh giá Decoder thích ứng
├── plot_b1.py               # Vẽ đồ thị Trade-off & lưới ảnh so sánh tái tạo
├── main.py                  # CLI pipeline: chạy đơn lẻ hoặc quét --sweep 3 mức sigma
├── test_step3.py            # Smoke test kiểm tra 100% pipeline
├── step3_colab.ipynb        # Notebook chạy trên Google Colab (GPU T4)
└── README.md                # Tài liệu hướng dẫn & phân tích lý thuyết
```

---

## 5. Hướng Dẫn Sử Dụng

### A. Chạy Kiểm Thử Đơn Vị (Local Smoke Test)
```bash
python step3/test_step3.py
```

### B. Chạy Huấn Luyện Mức Nhiễu Đơn Lẻ (ví dụ $\sigma = 0.5$)
```bash
python step3/main.py --sigma 0.5 --epochs 100 --decoder-epochs 30
```

### C. Chạy Quét Đa Mức Nhiễu ($\sigma \in \{0.1, 0.5, 1.0\}$)
```bash
python step3/main.py --sweep --sigmas 0.1,0.5,1.0 --epochs 100 --decoder-epochs 30
```
Lệnh trên sẽ tự động:
1. Huấn luyện SL và Decoder cho cả 3 mức $\sigma$.
2. In bảng so sánh Markdown tổng hợp.
3. Xuất file số liệu `results_b1.csv` và `results_b1.json`.
4. Xuất đồ thị đánh đổi `b1_tradeoff_curves.png`.
5. Xuất lưới so sánh chất lượng ảnh `b1_reconstruction_comparison.png`.

### D. Chạy trên Google Colab (GPU T4)
1. Tải notebook [step3_colab.ipynb](file:///d:/AbReTAPE/step3/step3_colab.ipynb) lên Google Colab.
2. Chọn **Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU**.
3. Chạy từng Cell theo thứ tự: Kết nối Google Drive, cài đặt thư viện phụ trợ, chạy kiểm thử và chạy quét đa mức $\sigma$. Kết quả và đồ thị sẽ tự động hiển thị trực quan và lưu trên Drive.

---

## 6. Kết Quả Kỳ Vọng (Privacy-Utility Trade-off)

| Mức nhiễu $\sigma$ | Test Accuracy (%) | PSNR (dB) | SSIM | Mức độ bảo vệ |
| :---: | :---: | :---: | :---: | :--- |
| **0.1** | $\sim 93.0\%$ | $\sim 24 - 27$ | $\sim 0.80 - 0.88$ | Rất thấp: Ảnh tái tạo rất nét |
| **0.5** | $\sim 89 - 91\%$ | $\sim 18 - 21$ | $\sim 0.60 - 0.70$ | Trung bình: Nhìn rõ vật thể, mờ chi tiết |
| **1.0** | $\sim 82 - 85\%$ | $\sim 14 - 17$ | $\sim 0.40 - 0.55$ | Giảm mạnh: Ảnh nhiều hạt nhiễu |

> **Điểm yếu then chốt cần nhấn mạnh trong Luận văn**:
> Khi tăng $\sigma$ từ 0.1 lên 1.0, độ chính xác phân loại sụt giảm nghiêm trọng (đánh đổi Utility lớn), nhưng ảnh tái tạo ở $\sigma = 1.0$ vẫn để lộ hình dáng đại thể của vật thể (xe hơi, máy bay, ngựa...). Đây là bằng chứng thực nghiệm rõ nhất cho thấy **Nhiễu Gaussian không phải là cơ chế phòng thủ tối ưu**, thúc đẩy việc nghiên cứu Encoder Task-Aware phi khả nghịch.
