# Bước 1 — Tấn Công Tái Tạo Bị Động (Passive Reconstruction Attack)

> **Đề tài**: *Absorption-Resistant Task-Aware Perceptual Encryption for Split Learning: Protecting Intermediate Representations against Reconstruction Attacks.*

---

## 1. Ý nghĩa của Bước 1 trong toàn bộ Đề tài
- **Xác nhận tiền đề cốt lõi**: Trong Vanilla Split Learning, Client gửi biểu diễn trung gian (IR / smashed data) $z = F_c(x)$ cho Server. Do không có cơ chế bảo vệ, thông tin cấu trúc không gian và màu sắc vẫn còn nguyên vẹn trong $z$. Bước 1 chứng minh lỗ hổng này thực sự tồn tại: **kẻ tấn công hoàn toàn có thể tái tạo lại ảnh gốc chỉ từ $z$**.
- **Thiết lập Attacker Baseline (Mức sàn cơ sở)**: Đây là mức sàn tham chiếu cao nhất của kẻ tấn công (về PSNR và SSIM) mà các cơ chế phòng thủ ở các bước sau (B2–B6) và Encoder đề xuất phải đánh bại (tức là làm suy giảm mạnh chất lượng tái tạo).
- **Cài đặt bộ Metric chuẩn hóa**: Thiết lập chuẩn các độ đo chất lượng ảnh: **PSNR**, **SSIM**, và **LPIPS** (trụ cột N3) để dùng xuyên suốt cho toàn bộ đề tài.
- **Sinh ra Decoder $D_\phi$**: Trọng số `b1_decoder.pt` sẽ được tái sử dụng trong các bài kiểm tra đánh giá thích ứng (Adaptive Evaluation ở Bước 5).

---

## 2. Mục tiêu & Tiêu chí đạt (Acceptance Criteria)
- **Mục tiêu**: Đóng băng mô hình Client $F_c$ từ Bước 0, huấn luyện Decoder $D_\phi$ ánh xạ $z \to \hat{x}$ theo bài toán hồi quy:
  $$\min_\phi \; \mathbb{E}\big[\|D_\phi(F_c(x)) - x\|^2\big]$$
- **Tiêu chí nghiệm thu**:
  - [x] Huấn luyện Decoder 30 epoch ổn định không lỗi.
  - [x] **PSNR > 20 dB** (kỳ vọng 25–30 dB với cut layer sau `layer1`).
  - [x] **SSIM > 0.60** (kỳ vọng 0.8+).
  - [x] Xuất ảnh đối chứng trực quan (`reconstruction_grid.png`) nhìn rõ đối tượng bằng mắt thường (đúng hình dáng, đúng lớp).
  - [x] Lưu checkpoint thành công: `b1_decoder.pt` và `best_b1_decoder.pt`.

---

## 3. Kiến trúc Mạng Decoder & Công thức
- **Kích thước IR đầu vào**: $z \in \mathbb{R}^{64 \times 32 \times 32}$ (đầu ra của `layer1` từ ResNet-18 Client).
- **Kiến trúc Decoder $D_\phi$**:
  - `Conv2d(64, 128, kernel=3, padding=1)` + `ReLU`
  - `Conv2d(128, 128, kernel=3, padding=1)` + `ReLU`
  - `Conv2d(128, 64, kernel=3, padding=1)` + `ReLU`
  - `Conv2d(64, 3, kernel=3, padding=1)`
- **Đầu ra**: Ảnh tái tạo $\hat{x} \in \mathbb{R}^{3 \times 32 \times 32}$.
- **Hàm mất mát**: $\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i=1}^B \|\hat{x}_i - x_i\|_2^2$.

---

## 4. Cấu trúc thư mục `step1/`
```text
step1/
├── decoder.py           # Kiến trúc mạng Decoder D_phi
├── metrics.py           # Bộ tính toán PSNR, SSIM, LPIPS và denormalize
├── attack.py            # Vòng lặp huấn luyện train_decoder_epoch & evaluate_attack
├── plot_attack.py       # Xuất ảnh lưới đối chứng và đồ thị chỉ số tấn công
├── main.py              # Script chạy chính: train 30 epoch, lưu checkpoint, history
├── test_step1.py        # Kiểm thử đơn vị (Smoke test) cho toàn bộ pipeline Bước 1
├── step1_colab.ipynb    # Jupyter Notebook chạy trên Google Colab
└── README.md            # Tài liệu hướng dẫn Bước 1
```

---

## 5. Hướng dẫn chạy

### A. Chạy trên máy Local / Server
```bash
# 1. Chạy kiểm thử đơn vị (Smoke test):
python step1/test_step1.py

# 2. Huấn luyện Decoder (30 epochs):
python step1/main.py --epochs 30 --lr 0.001 --eval-freq 5

# 3. Tiếp tục huấn luyện nếu bị ngắt (Resume):
python step1/main.py --resume

# 4. Tùy chỉnh đường dẫn checkpoint Bước 0:
python step1/main.py --client-ckpt step0/best_b0_vanilla.pt
```

### B. Chạy trên Google Colab (GPU T4)
1. Tải file `step1/step1_colab.ipynb` lên Google Colab.
2. Chọn môi trường thực thi: **Runtime** -> **Change runtime type** -> **T4 GPU**.
3. Chạy từng Cell theo thứ tự:
   - Mount Google Drive để lưu checkpoint vĩnh viễn: `/content/drive/MyDrive/AbReTAPE_Step1/`.
   - Cài đặt `scikit-image` và `lpips`.
   - Tải dataset và chạy smoke test.
   - Huấn luyện Decoder 30 epoch.
   - Hiển thị trực tiếp ảnh lưới so sánh `reconstruction_grid.png` và đồ thị `attack_curves.png`.

---

## 6. Xử lý Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **PSNR rất thấp (< 15 dB), ảnh tái tạo mờ mịt** | Quên nạp trọng số `b0_vanilla.pt` của Bước 0, hoặc Client chưa được đóng băng (`client.eval()`). | Kiểm tra tham số `--client-ckpt`, đảm bảo Client được nạp đúng weights đã train ở Bước 0. |
| **SSIM lỗi shape / axis** | Chưa denormalize ảnh về $[0, 1]$ hoặc sai trục channel. | Sử dụng hàm `psnr_ssim` trong `metrics.py` (đã cấu hình sẵn `channel_axis=2`). |
| **Loss MSE không giảm** | Tốc độ học quá lớn hoặc quá nhỏ. | Sử dụng Adam optimizer với learning rate `1e-3`. |
| **Thiếu module skimage** | Môi trường chưa cài `scikit-image`. | Cài đặt bằng `pip install scikit-image`. |
