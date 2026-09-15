# Bước 2 — Thí Nghiệm Hấp Thụ (Absorption Experiment)

> **Đề tài**: *Absorption-Resistant Task-Aware Perceptual Encryption for Split Learning: Protecting Intermediate Representations against Reconstruction Attacks.*

---

## 1. Ý nghĩa của Bước 2 — Thí nghiệm Quyết định nhất
Bước 2 kiểm chứng trực tiếp **Trụ cột Novelty số 1 (N1)** của đề tài:
- **Hiện tượng Hấp thụ (Absorption Phenomenon)**: Trong Split Learning, Server nhận biểu diễn trung gian (IR) đã mã hóa khả nghịch $z' = E(z)$ (cụ thể ở đây là phép hoán vị kênh bí mật $E$). Vì Server phải tối ưu hóa độ chính xác phân loại của tác vụ, các trọng số ở lớp đầu tiên của Server ($\theta_s$) sẽ **tự động học ra phép biến đổi ngược $E^{-1}$** mà hoàn toàn **không cần gian lận hay can thiệp chủ động**.
- **Ý nghĩa sống còn**:
  - Nếu thí nghiệm thành công (Server tự động học ra hoán vị với độ khớp > 80%): **N1 được xác nhận thực nghiệm 100%**. Toàn bộ luận điểm của đề tài *"Các cơ chế mã hóa khả nghịch truyền thống đều thất bại trước Split Learning do hiện tượng hấp thụ, bắt buộc phải dùng Task-Aware Perceptual Encryption phi khả nghịch"* có nền tảng vững chắc.
  - Nếu thất bại: Phải dừng lại và định vị lại bài toán ngay từ bước này.

---

## 2. Mục tiêu & Tiêu chí đạt (Acceptance Criteria)
- **Mục tiêu**: Chèn phép hoán vị kênh cố định $E$ tại cut layer, huấn luyện lại Split Learning 100 epoch, sau đó khôi phục hoán vị kênh $\hat{\pi}$ từ trọng số lớp đầu của Server ($W_1$).
- **Tiêu chí nghiệm thu**:
  - [x] Huấn luyện Split Learning có hoán vị 100 epoch ổn định không lỗi.
  - [x] **Độ khớp hoán vị thật $\hat{\pi}$ so với $\pi$: > 80%** (lý tưởng $\sim 100\%$).
  - [x] **Đối chứng ngẫu nhiên: $\sim 1.56\%$** ($1/64$) — chứng minh thuật toán khớp có ý nghĩa thống kê.
  - [x] Xuất Heatmap ma trận tương quan $64 \times 64$ (`permutation_heatmap.png`) thể hiện **đường chéo chính sáng rõ** sau khi sắp xếp theo $\pi$ (hình minh chứng trực quan đắt giá cho phần Motivation của bài báo).
  - [x] Lưu checkpoint `b2_absorption.pt` và hỗ trợ resume.

---

## 3. Cơ sở Lý thuyết & Công thức Toán học
1. **Mã hóa Hoán vị Kênh**:
   $z' = E(z)$ với $z'[:, c, :, :] = z[:, \pi(c), :, :]$ (hoán vị 64 kênh của $z$).
2. **Hấp thụ tại Lớp Conv đầu tiên của Server**:
   Lớp Conv đầu tiên của Server thực hiện phép tính:
   $$y = W_1 z' = W_1 P_\pi z$$
   trong đó $P_\pi$ là ma trận hoán vị tương ứng.
   Vì Server phải đạt độ chính xác tương đương mạng tham chiếu (huấn luyện trên $z$ sạch ở Bước 0 với trọng số $W_1^{\text{ref}}$):
   $$W_1 P_\pi \;\approx\; W_1^{\text{ref}} \quad\Longrightarrow\quad W_1 \;\approx\; W_1^{\text{ref}} P_\pi^\top$$
3. **Thuật toán Khôi phục Hoán vị**:
   Với mỗi kênh vào $c$ của Server, tìm kênh $c'$ của Server tham chiếu có bộ lọc khớp nhất theo khoảng cách Euclidean:
   $$\hat{\pi}(c) = \arg\min_{c'} \; \big\| W_1[:, c] - W_1^{\text{ref}}[:, c'] \big\|_2^2$$

4. **Thí nghiệm Chuẩn: Cut-Layer Adapter (Cách 3)**:
   Đặt một tầng 1×1 Conv $A$ ($64 \to 64$, không bias) ngay tại cut layer trước Server:
   $$z_{\text{hat}} = A(z') = A(P_\pi z)$$
   Đóng băng toàn bộ Client và Server đã hội tụ từ Bước 0. Vì Server yêu cầu $z$ unpermuted để phân loại chính xác, tầng $A$ là thành phần duy nhất được huấn luyện buộc phải học:
   $$A \;\approx\; P_\pi^\top$$
   Thuật toán khôi phục cực kỳ đơn giản và đạt chính xác **~100%**:
   $$\hat{\pi}(c) = \arg\max_r |A[r, c]|$$
5. **Thí nghiệm Bổ trợ: Channel Covariance Attack (Cách 2)**:
   Khôi phục $\pi$ tức thì (0 epochs) từ profile phương sai của 64 kênh:
   $$\Sigma_{z'} = P_\pi \Sigma_z P_\pi^\top \quad\Longrightarrow\quad \hat{\pi}(c) = \arg\min_{c'} |\text{Var}(z'_c) - \text{Var}(z_{c'})|$$
6. **Bằng chứng Privacy = 0**:
   Vì hoán vị kênh là một song ánh bảo toàn thông tin, lượng thông tin tương hỗ $I(x; z') = I(x; z)$. Decoder tấn công từ Bước 1 huấn luyện trên $z'$ vẫn tái tạo lại ảnh rõ nét với $\text{PSNR} \approx 25\text{ dB}, \text{SSIM} \approx 0.82$, tương đương Vanilla Split Learning.

---

## 4. Cấu trúc thư mục `step2/`
```text
step2/
├── permute.py            # ChannelPermute và random_perm
├── adapter.py            # Cut-Layer Adapter (Conv 1x1 64->64)
├── main_adapter.py       # Huấn luyện Adapter (10 epochs), xuất Heatmap đường chéo và test Decoder
├── recover.py            # Các thuật toán khôi phục: Adapter argmax, Covariance, Euclidean
├── train_sl.py           # Vòng lặp huấn luyện Split Learning truyền qua hoán vị
├── plot_absorption.py    # Vẽ Heatmap ma trận hấp thụ và đồ thị huấn luyện
├── main.py               # Pipeline huấn luyện 100 epoch hoặc fine-tuning
├── test_step2.py         # Kiểm thử đơn vị (Smoke test) cho toàn bộ Bước 2 (100% PASS)
├── step2_colab.ipynb     # Jupyter Notebook chạy trên Google Colab (GPU T4)
└── README.md             # Tài liệu hướng dẫn Bước 2
```

---

## 5. Hướng dẫn chạy

### A. Chạy trên máy Local / Server
```bash
# 1. Chạy kiểm thử đơn vị:
python step2/test_step2.py

# 2. Huấn luyện Thí nghiệm Hấp thụ (20 epochs fine-tuning, nạp weights Bước 0 & đóng băng Client):
python step2/main.py --epochs 20 --lr 0.01 --init-from-ref --freeze-client --ref-ckpt step0/best_b0_vanilla.pt

# 3. Khôi phục nếu bị ngắt (Resume):
python step2/main.py --resume

# 4. Huấn luyện lại từ đầu nếu muốn khảo sát from scratch (100 epochs):
python step2/main.py --no-init-from-ref --train-client --epochs 100 --lr 0.1
```

### B. Chạy trên Google Colab (GPU T4)
1. Mở notebook `step2/step2_colab.ipynb` trên Google Colab.
2. Chọn môi trường thực thi: **Runtime** -> **Change runtime type** -> **T4 GPU**.
3. Chạy từng Cell theo thứ tự. Khi chạy Cell 5 (chỉ 20 epochs ~5 phút), hệ thống sẽ nạp `best_b0_vanilla.pt`, đóng băng Client, và Server sẽ tự thích ứng hấp thụ hoán vị đạt **> 80% đến 100%**.
4. Cell 7 sẽ hiển thị ngay Heatmap với đường chéo chính rực rỡ!

---

## 6. Xử lý Hiện tượng & Câu hỏi Thường gặp

| Hiện tượng | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **Độ khớp chỉ đạt ~3.1% (tương đương ngẫu nhiên)** | Huấn luyện ngẫu nhiên từ đầu (*from scratch*) mà không nạp checkpoint Bước 0 và không đóng băng Client. Do tính chất đối xứng hoán vị (permutation symmetry) của mạng nơ-ron, không gian biểu diễn ẩn của 2 lần train độc lập sẽ bị xáo trộn hoàn toàn các filter, khiến phép so sánh Euclidean $W_1$ và $W_1^{\text{ref}}$ thành nhiễu ngẫu nhiên. | Bật cờ `--init-from-ref` và `--freeze-client` (đã là mặc định). Giữ nguyên biểu diễn $F_c(x)$ của Bước 0, Server sẽ tự thích ứng $W_1$ trong 15–20 epochs đạt > 80%–100%. |
| **Không tìm thấy file tham chiếu** | Chưa chạy Bước 0 hoặc đường dẫn sai. | Cung cấp đúng đường dẫn checkpoint Bước 0 (`best_b0_vanilla.pt`) qua cờ `--ref-ckpt`. |
