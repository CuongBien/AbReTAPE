# -*- coding: utf-8 -*-
# Module nạp dữ liệu Y tế cho Tầng 2 (Application Case Study): HAM10000 và PCAM
import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image

DEFAULT_NUM_WORKERS = 0 if sys.platform == "win32" else 2

# Thông số chuẩn hóa mặc định cho ảnh mô bệnh học (Histopathology / PCAM) và da liễu (Dermatology / HAM10000)
PCAM_MEAN = (0.7008, 0.5384, 0.6916)
PCAM_STD = (0.2350, 0.2774, 0.2129)

HAM10000_MEAN = (0.7630, 0.5456, 0.5700)
HAM10000_STD = (0.1409, 0.1526, 0.1699)


class RandomRotate90:
    """
    Xoay ngẫu nhiên bội số 90 độ (0, 90, 180, 270) cho ảnh mô học (Histopathology).
    Áp dụng được cho cả PIL Image, PyTorch Tensor và NumPy array.
    """
    def __call__(self, img):
        k = int(torch.randint(0, 4, (1,)).item())
        if k == 0:
            return img
        if isinstance(img, Image.Image):
            angles = [0, 90, 180, 270]
            return img.rotate(angles[k])
        elif isinstance(img, torch.Tensor):
            return torch.rot90(img, k, [-2, -1])
        elif isinstance(img, np.ndarray):
            return np.ascontiguousarray(np.rot90(img, k, axes=(0, 1) if img.ndim == 3 else (-2, -1)))
        return img


class PCAMNumpyDataset(Dataset):
    """
    Dataset loader nạp trực tiếp từ file NumPy (.npy) đã tiền xử lý:
    - Cắt giữa 64x64 từ 96x96 (bỏ 16px mỗi cạnh)
    - Định dạng uint8, hỗ trợ memory-mapping (mmap_mode='r')
    - Áp dụng augmentation chuẩn mô học: HorizontalFlip, VerticalFlip, Rotate 90
    """
    def __init__(self, images_path, labels_path, transform=None, mmap=True):
        if not os.path.isfile(images_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {images_path}")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Không tìm thấy file nhãn: {labels_path}")

        self.mmap = mmap
        self.images = np.load(images_path, mmap_mode="r" if mmap else None)
        self.labels = np.load(labels_path)
        self.transform = transform
        assert len(self.images) == len(self.labels), (
            f"Kích thước không khớp: {len(self.images)} ảnh vs {len(self.labels)} nhãn!"
        )

    def close(self):
        """Đóng mmap file để giải phóng file handle trên Windows."""
        if hasattr(self, "images") and hasattr(self.images, "_mmap") and self.images._mmap is not None:
            try:
                self.images._mmap.close()
            except Exception:
                pass
        self.images = None
        self.labels = None

    def __len__(self):
        return len(self.labels) if self.labels is not None else 0

    def __getitem__(self, idx):
        img_arr = self.images[idx]
        label = int(self.labels[idx])
        img = Image.fromarray(img_arr)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        return img, label


def get_pcam_dataloaders(data_dir="./data/pcam_processed", batch_size=128, num_workers=None, stats=None):
    """
    Nạp 3 tập dữ liệu PCAM từ mảng NumPy đã tiền xử lý:
    - Train: 50,000 ảnh phân tầng (25,000 class 0, 25,000 class 1)
    - Val: 5,000 ảnh phân tầng (2,500 class 0, 2,500 class 1)
    - Test: 32,768 ảnh đầy đủ
    - Kích thước ảnh: 64x64 (đã cắt giữa)
    - Augmentation: HorizontalFlip, VerticalFlip, Rotate 90 (bỏ random crop có padding)
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    mean, std = PCAM_MEAN, PCAM_STD
    stats_file = os.path.join(data_dir, "pcam_stats.json")
    if stats is not None:
        mean = tuple(stats.get("mean", PCAM_MEAN))
        std = tuple(stats.get("std", PCAM_STD))
    elif os.path.isfile(stats_file):
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                s = json.load(f)
                mean = tuple(s.get("mean", PCAM_MEAN))
                std = tuple(s.get("std", PCAM_STD))
        except Exception:
            pass

    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomRotate90(),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    train_img_p = os.path.join(data_dir, "train_images.npy")
    train_lbl_p = os.path.join(data_dir, "train_labels.npy")
    val_img_p = os.path.join(data_dir, "val_images.npy")
    val_lbl_p = os.path.join(data_dir, "val_labels.npy")
    test_img_p = os.path.join(data_dir, "test_images.npy")
    test_lbl_p = os.path.join(data_dir, "test_labels.npy")

    train_ds = PCAMNumpyDataset(train_img_p, train_lbl_p, transform=train_tf)
    val_ds = PCAMNumpyDataset(val_img_p, val_lbl_p, transform=eval_tf)
    test_ds = PCAMNumpyDataset(test_img_p, test_lbl_p, transform=eval_tf)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)

    meta = {
        "num_classes": 2,
        "in_channels": 3,
        "img_size": (64, 64),
        "mean": mean,
        "std": std,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds)
    }
    return train_loader, val_loader, test_loader, meta


class HAM10000Dataset(Dataset):
    """
    Dataset loader cho HAM10000 (Human Against Machine with 10000 training images):
    7 lớp: 0: akiec, 1: bcc, 2: bkl, 3: df, 4: mel, 5: nv, 6: vasc
    """
    LABEL_MAP = {
        'akiec': 0,  # Actinic keratoses and intraepithelial carcinoma
        'bcc': 1,    # Basal cell carcinoma
        'bkl': 2,    # Benign keratosis-like lesions
        'df': 3,     # Dermatofibroma
        'mel': 4,    # Melanoma
        'nv': 5,     # Melanocytic nevi
        'vasc': 6    # Vascular lesions
    }

    def __init__(self, data_dir, split="train", transform=None, img_size=(32, 32)):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.samples = []

        import pandas as pd
        csv_candidates = [
            os.path.join(data_dir, "HAM10000_metadata.csv"),
            os.path.join(data_dir, "metadata.csv"),
            os.path.join(data_dir, "ham10000.csv")
        ]
        csv_file = None
        for p in csv_candidates:
            if os.path.isfile(p):
                csv_file = p
                break

        if csv_file and os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
            img_dirs = [
                os.path.join(data_dir, "HAM10000_images_part_1"),
                os.path.join(data_dir, "HAM10000_images_part_2"),
                os.path.join(data_dir, "images"),
                data_dir
            ]

            if "split" in df.columns:
                sub_df = df[df["split"] == split]
            else:
                n_train = int(len(df) * 0.8)
                sub_df = df.iloc[:n_train] if split == "train" else df.iloc[n_train:]

            for _, row in sub_df.iterrows():
                img_id = str(row["image_id"])
                label_str = str(row["dx"]).lower()
                label = self.LABEL_MAP.get(label_str, 5)

                found_path = None
                for d in img_dirs:
                    candidate = os.path.join(d, f"{img_id}.jpg")
                    if os.path.isfile(candidate):
                        found_path = candidate
                        break

                if found_path:
                    self.samples.append((found_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.Resize(self.img_size)(img)
            img = T.ToTensor()(img)
            img = T.Normalize(HAM10000_MEAN, HAM10000_STD)(img)
        return img, label


def get_pcam(data_dir="./data/pcam", batch_size=128, img_size=(32, 32), num_workers=None):
    """
    Nạp dữ liệu PatchCamelyon (PCAM) thô qua torchvision.datasets.PCAM nếu chưa có.
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    train_tf = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(PCAM_MEAN, PCAM_STD)
    ])
    test_tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(PCAM_MEAN, PCAM_STD)
    ])

    try:
        trainset = torchvision.datasets.PCAM(root=data_dir, split="train", download=True, transform=train_tf)
        testset = torchvision.datasets.PCAM(root=data_dir, split="test", download=True, transform=test_tf)
    except Exception as e:
        print(f"[INFO] PCAM chưa sẵn sàng tải tự động: {e}. Vui lòng nạp thủ công vào thư mục {data_dir}")
        return None, None

    use_cuda = torch.cuda.is_available()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    return trainloader, testloader


def get_ham10000(data_dir="./data/ham10000", batch_size=128, img_size=(32, 32), num_workers=None):
    """
    Nạp dữ liệu HAM10000 đa lớp (7 lớp tổn thương da).
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    train_tf = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(HAM10000_MEAN, HAM10000_STD)
    ])
    test_tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(HAM10000_MEAN, HAM10000_STD)
    ])

    trainset = HAM10000Dataset(data_dir=data_dir, split="train", transform=train_tf, img_size=img_size)
    testset = HAM10000Dataset(data_dir=data_dir, split="test", transform=test_tf, img_size=img_size)

    if len(trainset) == 0:
        print(f"[INFO] Thư mục HAM10000 tại {data_dir} chưa có metadata/ảnh. Sẵn sàng khi bạn nạp dữ liệu.")
        return None, None

    use_cuda = torch.cuda.is_available()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    return trainloader, testloader
