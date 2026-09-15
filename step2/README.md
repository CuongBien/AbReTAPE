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

---

## 4. Cấu trúc thư mục `step2/`
```text
step2/
├── permute.py            # ChannelPermute và random_perm
├── recover.py            # Thuật toán khôi phục hoán vị recover_perm và match_accuracy
├── train_sl.py           # Vòng lặp huấn luyện Split Learning truyền qua hoán vị
├── plot_absorption.py    # Vẽ Heatmap ma trận hấp thụ 64x64 và đồ thị huấn luyện
├── main.py               # Pipeline huấn luyện 100 epoch, khôi phục và đối chứng
├── test_step2.py         # Kiểm thử đơn vị (Smoke test) cho toàn bộ Bước 2
├── step2_colab.ipynb     # Jupyter Notebook chạy trên Google Colab
└── README.md             # Tài liệu hướng dẫn Bước 2
```

---

## 5. Hướng dẫn chạy

### A. Chạy trên máy Local / Server
```bash
# 1. Chạy kiểm thử đơn vị:
python step2/test_step2.py

# 2. Huấn luyện Bước 2 (100 epochs, seed 42):
python step2/main.py --epochs 100 --batch-size 128 --perm-seed 42

# 3. Khôi phục nếu bị ngắt (Resume):
python step2/main.py --resume

# 4. Chỉ định đường dẫn checkpoint tham chiếu Bước 0:
python step2/main.py --ref-ckpt step0/b0_vanilla.pt
```

### B. Chạy trên Google Colab (GPU T4)
1. Tải notebook `step2/step2_colab.ipynb` lên Google Colab.
2. Chọn môi trường thực thi: **Runtime** -> **Change runtime type** -> **T4 GPU**.
3. Chạy từng Cell: Mount Google Drive để lưu checkpoint vĩnh viễn, nạp weights tham chiếu từ `AbReTAPE_Step0`, train 100 epoch, hiển thị Heatmap đường chéo chính xác nhận hiện tượng hấp thụ.

---

## 6. Xử lý Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **Độ khớp thấp (< 50%)** | Sai key state_dict hoặc so sánh sai lớp Conv. | Kiểm tra đúng key `layer2.0.conv1.weight` trong `b0_vanilla.pt`. |
| **Cả hoán vị thật lẫn ngẫu nhiên đều ~1.5%** | Chưa train đủ epoch, hoặc learning rate không đúng schedule. | Train đủ 100 epoch với MultiStepLR tại [50, 75]. |
| **Không tìm thấy file tham chiếu** | Chưa chạy Bước 0 hoặc đường dẫn sai. | Cung cấp đúng đường dẫn checkpoint Bước 0 qua cờ `--ref-ckpt`. |
