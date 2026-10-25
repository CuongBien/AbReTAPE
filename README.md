# AR-TAPE: Absorption-Resistant Task-Aware Perceptual Encoding for Split Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official research framework for **AR-TAPE (Absorption-Resistant Task-Aware Perceptual Encoding)** in Split Learning.

---

## 📌 Tổng Quan (Overview)

Trong kiến trúc **Split Learning (SL)**, mạng nơ-ron được phân tách giữa Client và Server tại lớp cắt (*cut-layer*). Dữ liệu trung gian (*smashed data* $Z = F_{\text{client}}(X)$) gửi sang Server thường chứa đựng toàn bộ cấu trúc hình học tri giác của ảnh gốc $X$.
- **Lỗ hổng cốt lõi:** Các cơ chế bảo vệ dựa trên phép biến đổi khả nghịch (affine/permutation) bị Server độc hại **tự động hấp thụ ánh xạ ngược** ($\theta_s \to E^{-1}$) thông qua quá trình lan truyền ngược gradient trong quá trình huấn luyện SL. Đồng thời, Server độc hại có thể thao túng gradient để chiếm đoạt không gian biểu diễn qua tấn công **FSHA (Feature Space Hijacking Attack)**.
- **Giải pháp AR-TAPE:** Phép chiếu trực giao thu gọn rank nhận biết tác vụ $\mathbf{P}_{\text{task}} = \mathbf{V}_{\text{task}}\mathbf{V}_{\text{task}}^\top$ kết hợp tối ưu hóa đối kháng Min-Max, triệt tiêu vĩnh viễn không gian tri giác dư thừa $\mathbf{P}_\perp z = 0$ và miễn nhiễm trước mọi kẻ tấn công bị động (Fresh MLP/Conv) lẫn chủ động (FSHA).

---

## 📊 Bảng Đối Chuẩn Master (Master Benchmark Results)

Dưới đây là bảng số liệu thực nghiệm chính thức trên **CIFAR-10 (ResNet-18 Cut-1)**, được trích xuất tự động 100% từ file JSON gốc với độ lệch chuẩn mẫu ($ddof=1$, $n=3$ seeds: $42, 7, 2024$):

| Phương Pháp | Bottleneck | $\lambda$ | Acc (10k Full) | Acc (2k Fast) | Fresh MLP (10k) | Fresh Conv (10k) | FSHA SSIM ($gs=5$) | FSHA Task Acc | $d\text{Cor}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 Vanilla** | Full ($D=65536$) | None | $94.71\%$ | N/A | N/A | $0.9955$ | $0.9057$ | $93.77\%$ | $0.7192$ |
| **B7 Mode F ($k=2048$)** | $k=2048$ | $0.0$ | $92.73\%$ | N/A | N/A | N/A | $0.0701$ | $73.81\%$ | N/A |
| **B7 Mode F ($k=1024$)** | $k=1024$ | $0.0$ | $91.53\%$ | N/A | $0.5461$ | $0.3597$ | $0.0668$ | $65.54\%$ | N/A |
| **B7 Mode F ($k=512$)** | $k=512$ | $0.0$ | $88.54\%$ | N/A | $0.4394$ | $0.2632$ | $0.0586$ | $68.00\%$ | N/A |
| **B8 Un-stabilized** | Full ($D=65536$) | $0.1$ | $92.38\%$ | $93.42\%$ | $0.4580$ | $0.7652$ | N/A | N/A | N/A |
| **B8 Stabilized** | Full ($D=65536$) | $0.1$ | $88.98\%$ | $93.53\%$ | $0.2621$ | **$0.6431$** | $0.0856$ | $44.34\%$ | $0.2244$ |
| **AR-TAPE v1 ($\lambda=0.0$)** | $k=2048$ | $0.0$ | $91.05 \pm 0.36\%$ | $92.78 \pm 0.13\%$ | $0.7077 \pm 0.0059$ | $0.5627 \pm 0.0065$ | $0.0402$ | $84.20\%$ | $0.7219$ |
| **AR-TAPE v1 ($\lambda=0.1$, s42)** | $k=2048$ | $0.1$ | $90.37\%$ | $92.69\%$ | $0.2261$ | $0.1938$ | N/A | N/A | $0.3020$ |
| **★ AR-TAPE v1 ($\lambda=0.25$, 3 seeds)** | $k=2048$ | $0.25$ | **$90.35 \pm 0.24\%$** | **$91.65 \pm 0.56\%$** | **$0.2159 \pm 0.0029$** | **$0.2014 \pm 0.0036$** | **$0.0384$** | **$84.25\%$** | **$0.2876 \pm 0.0077$** |
| **AR-TAPE v1 ($\lambda=0.5$, 3 seeds)** | $k=2048$ | $0.5$ | $89.41 \pm 0.50\%$ | $90.95 \pm 0.77\%$ | $0.2425 \pm 0.0144$ | $0.2315 \pm 0.0048$ | $0.0377$ | $83.44\%$ | $0.4348$ |

*(Xem thêm chi tiết tại [`master_benchmark_table.md`](master_benchmark_table.md) và các biểu đồ phân tích tại [`figures/`](figures/)).*

---

## 📁 Cấu Trúc Mã Nguồn (Repository Layout)

```
├── configs/                      # File cấu hình YAML chuẩn cho từng dataset & kịch bản
│   ├── cifar10_artape_k2048.yaml # Cấu hình chính thức điểm vận hành λ=0.25
│   ├── medical_ham10000.yaml     # Cấu hình bộ dữ liệu da liễu HAM10000
│   └── medical_pcam.yaml         # Cấu hình bộ dữ liệu mô bệnh học PCAM
│
├── src/                          # Thư viện lõi nghiên cứu
│   ├── data/                     # DataLoaders (CIFAR, HAM10000, PCAM, Factory)
│   ├── models/                   # Split ResNet & Bottleneck architectures
│   ├── defenses/                 # Các cơ chế phòng thủ (AR-TAPE, B1-B8)
│   ├── attacks/                  # Bộ công cụ tấn công (Fresh Inversion, FSHA, Co-adapted)
│   ├── metrics/                  # Bộ độ đo (Utility, Clinical, Security, Interpretability)
│   └── utils/                    # Checkpoint, trực quan hóa, logging
│
├── experiments/                  # Lưu trữ hồ sơ thực nghiệm
│   └── cifar10_poc/              # Hồ sơ đóng băng giai đoạn 1 (PoC CIFAR-10)
│       ├── legacy_scripts/       # Các script chạy từng bước trong quá trình nghiên cứu
│       └── README.md             # Hướng dẫn chi tiết tái lập lịch sử PoC
│
├── scripts/                      # Công cụ xuất báo cáo
│   ├── generate_master_table.py  # Sinh bảng Master trực tiếp từ JSON
│   ├── plot_pareto_frontier.py   # Vẽ đường cong Pareto Trade-off
│   └── plot_visual_comparison.py # Tạo bảng ảnh so sánh trực quan
│
├── train.py                      # [ENTRYPOINT 1] Huấn luyện Split Learning chuẩn
├── eval_attacks.py               # [ENTRYPOINT 2] Đánh giá khả năng phòng thủ trước đối thủ
├── benchmark.py                  # [ENTRYPOINT 3] Điều phối quy trình đa seed & sinh báo cáo
└── run_tests.py                  # Bộ kiểm thử đơn vị toàn diện (9/9 Unit Tests)
```

---

## ⚡ Hướng Dẫn Tái Lập Nhanh (Quickstart & Replication)

### 1. Cài đặt môi trường
```bash
# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### 2. Chạy kiểm thử đơn vị
Kiểm tra tính toàn vẹn của toàn bộ 9/9 modules:
```bash
python run_tests.py
```

### 3. Huấn luyện Split Learning với AR-TAPE
Chạy huấn luyện mô hình với điểm vận hành chuẩn $\lambda=0.25$:
```bash
python train.py \
    --dataset cifar10 \
    --defense ar_tape \
    --k 2048 \
    --lambda-val 0.25 \
    --epochs 100 \
    --seed 42 \
    --output-dir output/cifar10_ar_tape_lam0.25_s42
```
*(Thêm cờ `--dry-run` để kiểm thử nhanh 1 epoch).*

### 4. Đánh giá tính bảo mật trước các đòn tấn công
Đánh giá Fresh MLP và Fresh Conv trên checkpoint vừa huấn luyện:
```bash
python eval_attacks.py \
    --client-ckpt output/cifar10_ar_tape_lam0.25_s42/checkpoints/cifar10_ar_tape_k2048_lam0.25_s42_client.pt \
    --k 2048 \
    --epochs 30 \
    --output-dir output/cifar10_ar_tape_lam0.25_s42/eval_attacks
```

### 5. Tái sinh Bảng Master Benchmark
```bash
python scripts/generate_master_table.py
```
Lệnh này sẽ tự động đọc toàn bộ file JSON kết quả thô trong `output/`, tính toán độ lệch chuẩn mẫu $\text{ddof}=1$, và cập nhật `master_benchmark_table.md` cùng `master_benchmark_table.csv`.

---

## 🏥 Hướng Tới Domain Y Tế (Medical Domain Expansion)

Hệ thống đã sẵn sàng cho Tầng 2 với các bộ nạp dữ liệu và độ đo lâm sàng tích hợp sẵn:
- **Bộ dữ liệu:** [`HAM10000`](src/data/medical.py) (Dermatology đa lớp) và [`PCAM`](src/data/medical.py) (Histopathology nhị phân).
- **Độ đo lâm sàng:** Macro AUC-ROC, Macro F1, $\text{Sensitivity@90\%Specificity}$, và Ngưỡng tối ưu Youden's $J$.
- **Giải thích & Tin cậy:** Kiểm chứng Visual Attention Alignment qua Grad-CAM Heatmap Pearson correlation.
