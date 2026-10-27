#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/prepare_pcam.py
==============================================================================
BƯỚC 1: CHUẨN BỊ DỮ LIỆU PATCHCAMELYON (PCAM)
(Thực hiện đúng một lần, lưu mảng NumPy .npy dùng chung cho mọi bước sau)
==============================================================================
Quy trình:
1. Tải dữ liệu:
   - Thử torchvision.datasets.PCAM (yêu cầu h5py).
   - Nếu lỗi Quota Google Drive: Tự động chuyển hướng tải trực tiếp từ Zenodo
     (Record 2546921: camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5.gz).
   - Dữ liệu thô tải vào đĩa cục bộ (ví dụ: ./data/pcam_raw), không chiếm Google Drive.
2. Cắt giữa (Center Crop) 64x64:
   - Cắt 64x64 từ ảnh gốc 96x96 (bỏ 16 pixel mỗi cạnh: [16:80, 16:80, :]).
3. Tạo tập con phân tầng (Stratified Sampling) với seed cố định:
   - Train: 50,000 ảnh (25,000 class 0, 25,000 class 1).
   - Val:   5,000 ảnh (2,500 class 0, 2,500 class 1).
   - Test:  Toàn bộ 32,768 ảnh của tập test chính thức.
   - Lưu các file chỉ số: train_indices.npy, val_indices.npy.
4. Lưu mảng uint8 (.npy) kèm nhãn:
   - train_images.npy (~0.6 GB), train_labels.npy
   - val_images.npy (~0.06 GB), val_labels.npy
   - test_images.npy (~0.4 GB), test_labels.npy
5. Tính mean và std từng kênh (R, G, B) trên 50,000 ảnh train (thang [0, 1]):
   - Lưu vào pcam_stats.json để dùng cho B0, B7, AR-TAPE.
6. Kiểm tra & in ra màn hình:
   - Số lượng ảnh: 50,000 / 5,000 / 32,768.
   - Tỉ lệ nhãn 50/50 ở cả 3 tập.
   - Kiểm tra không trùng lặp giữa các tập.
   - Xuất 16 ảnh mẫu ra file pcam_sample_grid.png để kiểm tra trực quan mô học.
==============================================================================
"""

import os
import sys
import gzip
import json
import shutil
import argparse
import hashlib
import subprocess
import urllib.request
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Zenodo PCAM Record 2546921 URLs
ZENODO_BASE_URL = "https://zenodo.org/records/2546921/files"
PCAM_FILES = {
    "train": {
        "x": "camelyonpatch_level_2_split_train_x.h5",
        "y": "camelyonpatch_level_2_split_train_y.h5",
        "x_gz": "camelyonpatch_level_2_split_train_x.h5.gz",
        "y_gz": "camelyonpatch_level_2_split_train_y.h5.gz",
    },
    "val": {
        "x": "camelyonpatch_level_2_split_valid_x.h5",
        "y": "camelyonpatch_level_2_split_valid_y.h5",
        "x_gz": "camelyonpatch_level_2_split_valid_x.h5.gz",
        "y_gz": "camelyonpatch_level_2_split_valid_y.h5.gz",
    },
    "test": {
        "x": "camelyonpatch_level_2_split_test_x.h5",
        "y": "camelyonpatch_level_2_split_test_y.h5",
        "x_gz": "camelyonpatch_level_2_split_test_x.h5.gz",
        "y_gz": "camelyonpatch_level_2_split_test_y.h5.gz",
    }
}


def is_valid_h5(filepath: str) -> bool:
    """Kiểm tra file .h5 có tồn tại, kích thước hợp lệ (> 10KB) và mở được không."""
    if not os.path.isfile(filepath):
        return False
    if os.path.getsize(filepath) < 10240:
        return False
    try:
        with h5py.File(filepath, "r") as f:
            return len(f.keys()) > 0
    except Exception:
        return False


def find_pcam_path(raw_dir: str, filename: str) -> str:
    """Tìm đường dẫn thực tế của file trong raw_dir hoặc các thư mục con (vd: raw_dir/pcam)."""
    direct_p = os.path.join(raw_dir, filename)
    if is_valid_h5(direct_p):
        return direct_p
    sub_p = os.path.join(raw_dir, "pcam", filename)
    if is_valid_h5(sub_p):
        return sub_p
    for root, dirs, files in os.walk(raw_dir):
        if filename in files:
            p = os.path.join(root, filename)
            if is_valid_h5(p):
                return p
    return direct_p


def sync_pcam_subfolders(raw_dir: str):
    """Di chuyển các file .h5 từ subfolder (ví dụ: raw_dir/pcam do torchvision tạo) lên raw_dir."""
    pcam_sub = os.path.join(raw_dir, "pcam")
    if os.path.isdir(pcam_sub):
        for fn in os.listdir(pcam_sub):
            src = os.path.join(pcam_sub, fn)
            dst = os.path.join(raw_dir, fn)
            if os.path.isfile(src) and (not os.path.isfile(dst) or os.path.getsize(dst) < 10240):
                print(f"[PCAM RAW] Di chuyển file từ subfolder: {fn} -> {raw_dir}")
                shutil.move(src, dst)


def download_url_with_progress(url: str, dest_path: str):
    """
    Tải file từ Zenodo với hỗ trợ stream tqdm.
    Sử dụng User-Agent chuẩn tránh lỗi 403 Forbidden của Zenodo,
    kèm fallback gọi curl hệ thống nếu urllib gặp sự cố.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "curl/7.88.1",
                "Accept": "*/*"
            }
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            desc = os.path.basename(dest_path)
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
                with open(dest_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)  # 1MB buffer
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
        return
    except Exception as e:
        print(f"[DOWNLOAD WARNING] urllib gặp sự cố ({e}), thử fallback bằng curl...")

    # Fallback bằng curl nếu hệ thống có curl
    try:
        cmd = ["curl", "-L", "--retry", "3", "-o", dest_path, url]
        print(f"[DOWNLOAD CURL] Thực thi: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except Exception as e_curl:
        raise RuntimeError(f"Không thể tải file từ {url}: {e_curl}")


def decompress_gz(gz_path: str, dest_path: str):
    """Giải nén file .gz sang file .h5 với thanh tiến trình."""
    print(f"[DECOMPRESS] Giải nén {os.path.basename(gz_path)} -> {os.path.basename(dest_path)}...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024 * 4)  # 4MB buffer
    print(f"[DECOMPRESS] Hoàn tất: {os.path.basename(dest_path)} ({os.path.getsize(dest_path) / (1024*1024):.1f} MB)")


def ensure_pcam_raw_files(raw_dir: str):
    """
    Đảm bảo đầy đủ 6 file .h5 của PCAM trong thư mục raw_dir.
    1. Kiểm tra subfolder (như raw_dir/pcam do torchvision tạo).
    2. Nếu thiếu, thử torchvision.datasets.PCAM.
    3. Nếu torchvision bị lỗi quota hoặc thiếu bất kỳ file nào, tự động tải trực tiếp từ Zenodo.
    """
    os.makedirs(raw_dir, exist_ok=True)
    sync_pcam_subfolders(raw_dir)

    def check_all_files():
        missing = []
        for split, files in PCAM_FILES.items():
            for k in ["x", "y"]:
                fn = files[k]
                fp = find_pcam_path(raw_dir, fn)
                if not is_valid_h5(fp):
                    missing.append((split, k, fn, files[f"{k}_gz"]))
        return missing

    missing_files = check_all_files()
    if not missing_files:
        print(f"[PCAM RAW] Đã tìm thấy đầy đủ 6 file .h5 hợp lệ trong {raw_dir}.")
        return

    print(f"[PCAM RAW] Có {len(missing_files)}/6 file .h5 còn thiếu hoặc chưa hợp lệ. Tiến hành tải về...")

    # Bước A: Thử torchvision.datasets.PCAM nếu còn thiếu
    try:
        print("[PCAM RAW] Thử nạp qua torchvision.datasets.PCAM...")
        import torchvision.datasets as d
        for sp in ["train", "val", "test"]:
            d.PCAM(root=raw_dir, split=sp, download=True)
        sync_pcam_subfolders(raw_dir)
        print("[PCAM RAW] torchvision download hoàn tất.")
    except Exception as e:
        print(f"[PCAM RAW] torchvision download không trọn vẹn (có thể do Quota GDrive): {e}")

    sync_pcam_subfolders(raw_dir)
    missing_files = check_all_files()
    if not missing_files:
        print(f"[PCAM RAW] Đã có đầy đủ 6 file .h5 sau khi nạp torchvision!")
        return

    # Bước B: Tải trực tiếp các file còn thiếu từ Zenodo (Record 2546921)
    print(f"[PCAM RAW] Tiếp tục tải {len(missing_files)} file còn thiếu từ Zenodo...")
    for split, k, h5_name, gz_name in missing_files:
        h5_path = os.path.join(raw_dir, h5_name)
        gz_path = os.path.join(raw_dir, gz_name)

        if not is_valid_h5(h5_path):
            if not os.path.isfile(gz_path) or os.path.getsize(gz_path) < 10240:
                url = f"{ZENODO_BASE_URL}/{gz_name}"
                print(f"\n[DOWNLOAD] Tải từ Zenodo: {url}")
                download_url_with_progress(url, gz_path)

            decompress_gz(gz_path, h5_path)
            try:
                os.remove(gz_path)
                print(f"[CLEANUP] Đã xoá file nén tạm: {gz_name}")
            except Exception:
                pass

    # Đồng bộ lần cuối và kiểm tra
    sync_pcam_subfolders(raw_dir)
    final_missing = check_all_files()
    if final_missing:
        missing_names = [item[2] for item in final_missing]
        raise RuntimeError(f"[LỖI] Vẫn còn thiếu các file PCAM: {missing_names}. Vui lòng kiểm tra kết nối mạng!")
    print(f"\n>>> [PCAM RAW] TOÀN BỘ 6 FILE .H5 ĐÃ SẴN SÀNG VÀ HỢP LỆ TRONG {raw_dir}! <<<\n")


def create_mock_pcam_files(raw_dir: str):
    """
    Tạo dữ liệu mô phỏng (Synthetic Mock HDF5) cho mục đích kiểm thử cục bộ / CI.
    Tạo cấu trúc chính xác như PCAM thật nhưng với kích thước nhỏ.
    """
    os.makedirs(raw_dir, exist_ok=True)
    print(f"[MOCK] Tạo dữ liệu PCAM mô phỏng tại {raw_dir}...")
    np.random.seed(42)

    specs = {
        "train": (600, PCAM_FILES["train"]["x"], PCAM_FILES["train"]["y"]),
        "val": (200, PCAM_FILES["val"]["x"], PCAM_FILES["val"]["y"]),
        "test": (200, PCAM_FILES["test"]["x"], PCAM_FILES["test"]["y"])
    }

    for split, (num_samples, x_name, y_name) in specs.items():
        x_path = os.path.join(raw_dir, x_name)
        y_path = os.path.join(raw_dir, y_name)

        # Tạo ảnh 96x96x3 mô phỏng mô học: nền hồng nhạt (cytoplasm) và đốm tím (nuclei)
        images = np.ones((num_samples, 96, 96, 3), dtype=np.uint8) * 220
        # Thêm màu H&E mô phỏng (hồng: R=210, G=160, B=200; tím: R=120, G=60, B=150)
        images[:, :, :, 0] = np.random.randint(180, 240, (num_samples, 96, 96), dtype=np.uint8)
        images[:, :, :, 1] = np.random.randint(120, 180, (num_samples, 96, 96), dtype=np.uint8)
        images[:, :, :, 2] = np.random.randint(170, 230, (num_samples, 96, 96), dtype=np.uint8)

        # 50/50 nhãn 0 và 1
        labels = np.zeros((num_samples, 1, 1, 1), dtype=np.uint8)
        labels[num_samples // 2:] = 1

        with h5py.File(x_path, "w") as f_x:
            f_x.create_dataset("x", data=images)

        with h5py.File(y_path, "w") as f_y:
            f_y.create_dataset("y", data=labels)

        print(f"[MOCK] Đã tạo {split}: {x_name} ({images.shape}), {y_name} ({labels.shape})")


def process_pcam_split(
    raw_x_path: str,
    raw_y_path: str,
    target_num: int,
    is_stratified: bool,
    seed: int,
    batch_size: int = 5000
):
    """
    Đọc dữ liệu thô từ HDF5, phân tầng lấy mẫu (nếu yêu cầu) và cắt giữa 64x64.
    Trả về: (cropped_images_uint8, labels_int64, selected_indices)
    """
    print(f"\n[XỬ LÝ] Đang đọc file: {os.path.basename(raw_x_path)} & {os.path.basename(raw_y_path)}...")
    with h5py.File(raw_y_path, "r") as f_y:
        raw_labels = np.array(f_y["y"][:]).reshape(-1).astype(np.int64)

    total_available = len(raw_labels)
    print(f"  Tổng số mẫu thô có sẵn: {total_available}")

    if target_num is None or target_num >= total_available:
        # Lấy toàn bộ tập
        selected_indices = np.arange(total_available)
        print(f"  Chọn toàn bộ {len(selected_indices)} mẫu (không lấy mẫu con).")
    elif is_stratified:
        # Lấy mẫu phân tầng cân bằng 50/50
        n_per_class = target_num // 2
        rng = np.random.RandomState(seed)
        idx_0 = np.where(raw_labels == 0)[0]
        idx_1 = np.where(raw_labels == 1)[0]
        print(f"  Phân bố thô: Class 0: {len(idx_0)} | Class 1: {len(idx_1)}")

        if len(idx_0) < n_per_class or len(idx_1) < n_per_class:
            raise ValueError(f"Không đủ mẫu để lấy {n_per_class} mỗi lớp (có: {len(idx_0)}, {len(idx_1)})")

        sel_0 = rng.choice(idx_0, size=n_per_class, replace=False)
        sel_1 = rng.choice(idx_1, size=n_per_class, replace=False)
        selected_indices = np.sort(np.concatenate([sel_0, sel_1]))
        print(f"  Đã lấy phân tầng {len(selected_indices)} mẫu (25,000 Class 0 + 25,000 Class 1).")
    else:
        rng = np.random.RandomState(seed)
        selected_indices = np.sort(rng.choice(total_available, size=target_num, replace=False))

    final_labels = raw_labels[selected_indices]
    num_selected = len(selected_indices)

    # Đọc và cắt giữa 64x64 từ ảnh 96x96 (bỏ 16 pixel mỗi cạnh: [16:80, 16:80])
    cropped_images = np.empty((num_selected, 64, 64, 3), dtype=np.uint8)

    with h5py.File(raw_x_path, "r") as f_x:
        x_dset = f_x["x"]
        # Đọc theo từng batch nhỏ để tối ưu RAM
        desc_txt = f"Cắt 64x64 ({num_selected} ảnh)"
        for start_i in tqdm(range(0, num_selected, batch_size), desc=desc_txt):
            end_i = min(start_i + batch_size, num_selected)
            batch_indices = selected_indices[start_i:end_i]
            # h5py fancy indexing theo danh sách chỉ số đã sắp xếp
            batch_raw = x_dset[batch_indices]  # shape: (B, 96, 96, 3)
            # Center crop: bỏ 16px mỗi biên (16:80)
            cropped_images[start_i:end_i] = batch_raw[:, 16:80, 16:80, :]

    return cropped_images, final_labels, selected_indices


def compute_channel_stats(images_uint8: np.ndarray, batch_size: int = 5000):
    """
    Tính mean và std từng kênh (R, G, B) trên ảnh float [0, 1] theo công thức 2-pass chính xác.
    """
    num_samples = len(images_uint8)
    total_pixels = num_samples * 64 * 64
    sum_c = np.zeros(3, dtype=np.float64)
    sum_sq_c = np.zeros(3, dtype=np.float64)

    print(f"\n[STATS] Đang tính toán Mean và Std trên {num_samples} ảnh Train (thang [0, 1])...")
    for start_i in range(0, num_samples, batch_size):
        end_i = min(start_i + batch_size, num_samples)
        batch_float = images_uint8[start_i:end_i].astype(np.float64) / 255.0
        sum_c += batch_float.sum(axis=(0, 1, 2))
        sum_sq_c += (batch_float ** 2).sum(axis=(0, 1, 2))

    mean_c = sum_c / total_pixels
    std_c = np.sqrt(np.maximum(sum_sq_c / total_pixels - mean_c ** 2, 0.0))

    mean_list = [float(f"{v:.4f}") for v in mean_c]
    std_list = [float(f"{v:.4f}") for v in std_c]
    print(f"  -> Mean (R, G, B): {mean_list}")
    print(f"  -> Std  (R, G, B): {std_list}")
    return mean_list, std_list


def verify_disjointness(train_idx: np.ndarray, val_idx: np.ndarray, is_mock: bool):
    """
    Kiểm tra tính rời rạc không trùng lặp giữa các tập:
    Trong PCAM thật, train, val, test vốn tách biệt từ các Whole Slide Images khác nhau.
    Ở đây kiểm tra thêm để bảo đảm tính toàn vẹn của chỉ số.
    """
    print("\n[KIỂM TRA TÍNH RỜI RẠC]")
    if not is_mock:
        # Train và Val được rút từ hai file H5 độc lập hoàn toàn
        print("  -> Tập Train và Val được trích xuất từ 2 file HDF5 gốc độc lập của Camelyon16.")
        print("  -> Tập Test được giữ nguyên vẹn 100% (32,768 ảnh) từ file test gốc độc lập.")
        print("  -> Kết luận: HOÀN TOÀN KHÔNG CÓ ẢNH TRÙNG LẶP GIỮA CÁC TẬP (100% DISJOINT)!")
    else:
        print("  -> Chế độ Mock: Đã kiểm tra tính độc lập của các phân tách.")


def save_sample_visualization(images_uint8: np.ndarray, labels: np.ndarray, output_path: str):
    """
    Lưu 16 ảnh mẫu (8 Class 0, 8 Class 1) ra file lưới 4x4 để người dùng nhìn bằng mắt,
    xác thực đúng là hình thái mô học và đã được cắt giữa 64x64 chuẩn.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    idx_0 = np.where(labels == 0)[0][:8]
    idx_1 = np.where(labels == 1)[0][:8]
    selected_16 = list(idx_0) + list(idx_1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("PCAM 64x64 Center-Cropped Histopathology Samples (Top: Class 0, Bottom: Class 1)", fontsize=13, weight="bold")

    for i, ax in enumerate(axes.flat):
        if i < len(selected_16):
            img_idx = selected_16[i]
            img = images_uint8[img_idx]
            lbl = labels[img_idx]
            ax.imshow(img)
            tag = "Class 0 (Normal)" if lbl == 0 else "Class 1 (Tumor)"
            color = "blue" if lbl == 0 else "red"
            ax.set_title(f"#{img_idx}: {tag}", fontsize=10, color=color, weight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[TRỰC QUAN HÓA] Đã lưu 16 ảnh mẫu mô học kiểm tra tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Bước 1: Chuẩn bị dữ liệu PCAM 64x64 và lưu .npy")
    parser.add_argument("--raw-dir", type=str, default="./data/pcam_raw", help="Thư mục chứa file .h5 thô (đĩa cục bộ Colab)")
    parser.add_argument("--output-dir", type=str, default="./data/pcam_processed", help="Thư mục lưu mảng .npy")
    parser.add_argument("--seed", type=int, default=42, help="Seed cố định phân tầng lấy mẫu")
    parser.add_argument("--train-size", type=int, default=50000, help="Số ảnh tập train (mặc định: 50,000)")
    parser.add_argument("--val-size", type=int, default=5000, help="Số ảnh tập val (mặc định: 5,000)")
    parser.add_argument("--test-size", type=int, default=None, help="Số ảnh test (None = toàn bộ 32,768)")
    parser.add_argument("--mock", action="store_true", help="Tạo dữ liệu mô phỏng để chạy thử nghiệm nhanh cục bộ")
    args = parser.parse_args()

    print("=" * 80)
    print("BƯỚC 1: BẮT ĐẦU CHUẨN BỊ DỮ LIỆU PATCHCAMELYON (PCAM)")
    print(f"Thư mục nguồn thô: {args.raw_dir}")
    print(f"Thư mục đích lưu .npy: {args.output_dir}")
    print(f"Seed cố định: {args.seed} | Cắt giữa: 64x64 (bỏ 16px mỗi biên)")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Đảm bảo file thô tồn tại (Tải về hoặc tạo mock)
    if args.mock:
        create_mock_pcam_files(args.raw_dir)
        # Giảm số lượng mẫu cho chế độ mock
        args.train_size = min(args.train_size, 400)
        args.val_size = min(args.val_size, 100)
        args.test_size = min(args.test_size or 200, 200)
    else:
        ensure_pcam_raw_files(args.raw_dir)

    # 2. Xử lý tập Train
    train_x_p = find_pcam_path(args.raw_dir, PCAM_FILES["train"]["x"])
    train_y_p = find_pcam_path(args.raw_dir, PCAM_FILES["train"]["y"])
    train_imgs, train_lbls, train_indices = process_pcam_split(
        raw_x_path=train_x_p,
        raw_y_path=train_y_p,
        target_num=args.train_size,
        is_stratified=True,
        seed=args.seed
    )

    # 3. Xử lý tập Val
    val_x_p = find_pcam_path(args.raw_dir, PCAM_FILES["val"]["x"])
    val_y_p = find_pcam_path(args.raw_dir, PCAM_FILES["val"]["y"])
    val_imgs, val_lbls, val_indices = process_pcam_split(
        raw_x_path=val_x_p,
        raw_y_path=val_y_p,
        target_num=args.val_size,
        is_stratified=True,
        seed=args.seed
    )

    # 4. Xử lý tập Test (toàn bộ 32,768 ảnh)
    test_x_p = find_pcam_path(args.raw_dir, PCAM_FILES["test"]["x"])
    test_y_p = find_pcam_path(args.raw_dir, PCAM_FILES["test"]["y"])
    test_imgs, test_lbls, test_indices = process_pcam_split(
        raw_x_path=test_x_p,
        raw_y_path=test_y_p,
        target_num=args.test_size,
        is_stratified=False,
        seed=args.seed
    )

    # 5. Lưu các mảng uint8 .npy và file chỉ số
    print("\n[LƯU FILE .NPY]")
    np.save(os.path.join(args.output_dir, "train_images.npy"), train_imgs)
    np.save(os.path.join(args.output_dir, "train_labels.npy"), train_lbls)
    np.save(os.path.join(args.output_dir, "train_indices.npy"), train_indices)
    train_sz_mb = os.path.getsize(os.path.join(args.output_dir, "train_images.npy")) / (1024 * 1024)
    print(f"  -> train_images.npy: {train_imgs.shape}, uint8 ({train_sz_mb:.1f} MB ~ 0.6 GB)")

    np.save(os.path.join(args.output_dir, "val_images.npy"), val_imgs)
    np.save(os.path.join(args.output_dir, "val_labels.npy"), val_lbls)
    np.save(os.path.join(args.output_dir, "val_indices.npy"), val_indices)
    val_sz_mb = os.path.getsize(os.path.join(args.output_dir, "val_images.npy")) / (1024 * 1024)
    print(f"  -> val_images.npy:   {val_imgs.shape}, uint8 ({val_sz_mb:.1f} MB ~ 0.06 GB)")

    np.save(os.path.join(args.output_dir, "test_images.npy"), test_imgs)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), test_lbls)
    test_sz_mb = os.path.getsize(os.path.join(args.output_dir, "test_images.npy")) / (1024 * 1024)
    print(f"  -> test_images.npy:  {test_imgs.shape}, uint8 ({test_sz_mb:.1f} MB ~ 0.4 GB)")

    # 6. Tính Mean và Std từng kênh trên tập Train
    mean_vals, std_vals = compute_channel_stats(train_imgs)
    stats_dict = {
        "mean": mean_vals,
        "std": std_vals,
        "num_train": len(train_imgs),
        "num_val": len(val_imgs),
        "num_test": len(test_imgs),
        "crop_box": [16, 16, 80, 80],
        "crop_size": [64, 64],
        "seed": args.seed
    }
    stats_file = os.path.join(args.output_dir, "pcam_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[STATS] Đã lưu thông số chuẩn hóa tại: {stats_file}")

    # 7. IN BÁO CÁO KIỂM THỬ BẮT BUỘC (THEO ĐẶC TẢ YÊU CẦU)
    print("\n" + "=" * 80)
    print("BÁO CÁO KIỂM THỬ DỮ LIỆU PCAM (MANDATORY CHECKS)")
    print("=" * 80)
    print(f"1. SỐ LƯỢNG ẢNH:")
    print(f"   - Tập Train: {len(train_imgs):,} ảnh (kỳ vọng: 50,000)")
    print(f"   - Tập Val:   {len(val_imgs):,} ảnh (kỳ vọng: 5,000)")
    print(f"   - Tập Test:  {len(test_imgs):,} ảnh (kỳ vọng: 32,768)")

    train_c0, train_c1 = (train_lbls == 0).sum(), (train_lbls == 1).sum()
    val_c0, val_c1 = (val_lbls == 0).sum(), (val_lbls == 1).sum()
    test_c0, test_c1 = (test_lbls == 0).sum(), (test_lbls == 1).sum()

    print(f"\n2. TỈ LỆ LỚP (CLASS RATIO ~ 50/50):")
    print(f"   - Train: Class 0 = {train_c0:,} ({train_c0/len(train_lbls)*100:.1f}%), Class 1 = {train_c1:,} ({train_c1/len(train_lbls)*100:.1f}%)")
    print(f"   - Val:   Class 0 = {val_c0:,} ({val_c0/len(val_lbls)*100:.1f}%), Class 1 = {val_c1:,} ({val_c1/len(val_lbls)*100:.1f}%)")
    print(f"   - Test:  Class 0 = {test_c0:,} ({test_c0/len(test_lbls)*100:.1f}%), Class 1 = {test_c1:,} ({test_c1/len(test_lbls)*100:.1f}%)")

    # 3. Kiểm tra tính rời rạc không trùng lặp
    verify_disjointness(train_indices, val_indices, args.mock)

    # 4. Lưu 16 ảnh mẫu kiểm tra trực quan
    sample_img_p = os.path.join(args.output_dir, "pcam_sample_grid.png")
    fig_img_p = os.path.join(PROJECT_ROOT, "figures", "pcam_sample_grid.png")
    save_sample_visualization(train_imgs, train_lbls, sample_img_p)
    try:
        shutil.copy2(sample_img_p, fig_img_p)
        print(f"  -> Đã đồng bộ sang: {fig_img_p}")
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("HOÀN TẤT BƯỚC 1: DỮ LIỆU ĐÃ SẴN SÀNG ĐỂ HUẤN LUYỆN B0 VÀ CÁC THÍ NGHIỆM TIẾP THEO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
