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

## 4. Cấu trúc thư mục
```text
step0/
├── model.py            # ClientModel, ServerModel, resnet18_cifar
├── data.py             # DataLoader nạp và tiền xử lý CIFAR-10
├── train.py            # train_epoch (SL vòng lặp), evaluate
├── main.py             # Huấn luyện Vanilla SL (100 epoch), lưu b0_vanilla.pt
├── centralized.py      # Script Centralized Training đối chứng
├── download_cifar.py   # Script tải dữ liệu CIFAR-10 đa luồng có resume
└── README.md           # Tài liệu hướng dẫn Bước 0
```

## 5. Hướng dẫn chạy
```bash
# Chạy huấn luyện Vanilla Split Learning:
python step0/main.py

# Hoặc chạy đối chứng Centralized Training:
python step0/centralized.py
```
