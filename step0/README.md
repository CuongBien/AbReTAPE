# Bước 0 — Vanilla Split Learning (CIFAR-10, ResNet-18)

## 1. Mục tiêu & Tiêu chí đạt
- **Mục tiêu**: Xây dựng nền tảng Vanilla Split Learning (SL) 2 bên (Client – Server) trên CIFAR-10 với kiến trúc ResNet-18 (đơn tiến trình).
- **Tiêu chí nghiệm thu (Acceptance Criteria)**:
  - [x] Huấn luyện 100 epoch không lỗi.
  - [x] Test accuracy sau 100 epoch đạt **≥ 92%**.
  - [x] SL accuracy tương đương Centralized accuracy (độ chênh lệch < 1–2%).
  - [x] Lưu thành công checkpoint `b0_vanilla.pt` chứa trọng số của cả Client (`client`) và Server (`server`).

## 2. Kiến trúc mạng & Điểm cắt (Cut Layer)
- **Kiến trúc cơ sở**: ResNet-18 điều chỉnh cho CIFAR-10 (ảnh kích thước $32 \times 32$):
  - `conv1`: kernel $3 \times 3$, stride 1, padding 1 (thay cho $7 \times 7$ stride 2).
  - `maxpool`: thay bằng `nn.Identity()` để bảo toàn độ phân giải ban đầu.
  - `fc`: phân loại 10 lớp (`num_classes=10`).
- **Phân chia Client – Server**:
  - **Client**: Stem (`conv1` + `bn1` + `relu`) + `layer1`.
    - Đầu ra là biểu diễn trung gian (IR / smashed data): $z \in \mathbb{R}^{64 \times 32 \times 32}$.
  - **Server**: `layer2` + `layer3` + `layer4` + `avgpool` + `fc`.
    - Tiếp nhận $z$, tiếp tục lan truyền xuôi và tính toán hàm mất mát CrossEntropy.

## 3. Vòng lặp lan truyền ngược qua biên giới (Backpropagation across boundary)
Theo cơ chế chuẩn của Vepakomma et al. (2018):
```python
# Forward:
z = client(x)                          # IR, giữ computation graph ở client
z_d = z.detach().requires_grad_(True)  # Cắt đồ thị phía server
logits = server(z_d)
loss = criterion(logits, y)

# Backward Server:
loss.backward()                        # Tính gradient dL/dz tại z_d.grad
opt_s.step()

# Backward Client:
z.backward(z_d.grad)                   # Truyền dL/dz ngược qua biên giới client-server
opt_c.step()
```

## 4. Cơ chế lưu Checkpoint, Lịch sử & Trực quan hóa (Visualization)
- **Lưu Best Model (`best_b0_vanilla.pt`)**: Tự động lưu khi mô hình đạt test accuracy cao nhất, bao gồm trọng số mô hình, trạng thái optimizer, scheduler, best accuracy và toàn bộ lịch sử huấn luyện.
- **Lưu Checkpoint ngắt quãng (`last_checkpoint.pt`)**: Tự động lưu sau mỗi epoch bao gồm cả trạng thái `opt_c`, `opt_s`, `sched_c`, `sched_s`, `history` để có thể tiếp tục train bất kỳ lúc nào nếu bị gián đoạn.
- **Lưu Reference Model (`b0_vanilla.pt`)**: Trọng số tham chiếu cố định khi hoàn thành đủ 100 epoch (dùng cho Bước 2 — Thí nghiệm hấp thụ).
- **Lưu Lịch sử Huấn luyện (`history.json` & `history.csv`)**: Ghi nhận chi tiết từng epoch: `epoch`, `train_loss`, `train_acc`, `test_loss`, `test_acc`, `lr`, `epoch_time`. Tự động khôi phục và nối tiếp khi dùng `--resume`.
- **Trực quan hóa Đồ thị (`training_curves.png`)**: Tự động vẽ và cập nhật biểu đồ 4 thông số: Loss curves (Train vs Test), Accuracy curves (Train vs Test với mốc Best Acc), Learning Rate schedule, và thời gian huấn luyện mỗi epoch.

## 5. Cấu trúc thư mục
```text
step0/
├── model.py            # ClientModel, ServerModel, resnet18_cifar
├── data.py             # DataLoader nạp và tiền xử lý CIFAR-10
├── train.py            # train_epoch (tính loss & acc), evaluate (tính loss & acc)
├── plot.py             # Module vẽ đồ thị trực quan hóa quá trình huấn luyện
├── main.py             # Huấn luyện Vanilla SL, lưu checkpoint, history, xuất đồ thị
├── centralized.py      # Script Centralized Training đối chứng (có checkpoint & đồ thị)
├── download_cifar.py   # Script tải dữ liệu CIFAR-10 đa luồng có resume
├── test_step0.py       # Kiểm thử đơn vị (Smoke test) cho toàn bộ pipeline
├── train_colab.ipynb   # Jupyter Notebook chạy huấn luyện trên Google Colab
└── README.md           # Tài liệu hướng dẫn Bước 0
```

## 6. Hướng dẫn chạy

### A. Chạy trên máy Local (hoặc Server / GPU)
```bash
# 1. Kiểm tra đơn vị (Smoke test):
python step0/test_step0.py

# 2. Tải dữ liệu CIFAR-10 siêu tốc:
python step0/download_cifar.py

# 3. Chạy huấn luyện Vanilla Split Learning (100 epochs, eval mỗi epoch):
python step0/main.py --epochs 100 --batch-size 128 --eval-freq 1

# 4. Khôi phục huấn luyện tiếp từ checkpoint gần nhất nếu bị ngắt:
python step0/main.py --resume

# 5. Vẽ lại đồ thị từ file history.json bất kỳ lúc nào:
python step0/plot.py --history step0/history.json --output step0/training_curves.png

# 6. Chạy đối chứng Centralized Training:
python step0/centralized.py --epochs 100 --eval-freq 1
```

### B. Chạy trên Google Colab (Khuyên dùng T4 GPU miễn phí)
1. Tải notebook `train_colab.ipynb` lên Google Colab hoặc kết nối trực tiếp với GitHub Repository.
2. Đổi môi trường thực thi sang GPU: **Runtime** -> **Change runtime type** -> Chọn **T4 GPU** -> Nhấn **Save**.
3. Chạy từng Cell theo thứ tự:
   - **Cell 1 & 2**: Kiểm tra GPU (`nvidia-smi`).
   - **Cell 3**: Mount Google Drive để lưu checkpoint vĩnh viễn (tránh bị mất khi Colab timeout).
   - **Cell 4**: Thiết lập thư mục làm việc và tải dữ liệu `download_cifar.py`.
   - **Cell 5 & 6**: Chạy Smoke test và bắt đầu train Split Learning. Checkpoint và đồ thị được lưu trực tiếp vào Drive `/content/drive/MyDrive/AbReTAPE_Step0/`.
   - **Cell 7**: Dùng để Resume nếu phiên Colab bị gián đoạn.
   - **Cell 8 & 9**: Hiển thị đồ thị trực quan và bảng tổng kết Best Accuracy.

