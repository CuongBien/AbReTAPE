# -*- coding: utf-8 -*-
"""
src/data/factory.py
Bộ nạp dữ liệu thống nhất (Unified Data Factory) cho cả CIFAR và Y tế (HAM10000, PCAM).
"""
import os
import sys
import torch
from torch.utils.data import DataLoader, Subset

from .cifar import (
    get_cifar10,
    get_cifar100,
    get_cifar10_fsha_splits,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD
)
from .medical import (
    get_pcam,
    get_pcam_dataloaders,
    get_ham10000,
    PCAM_MEAN,
    PCAM_STD,
    HAM10000_MEAN,
    HAM10000_STD
)


DATASET_METADATA = {
    "cifar10": {
        "num_classes": 10,
        "in_channels": 3,
        "img_size": (32, 32),
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD
    },
    "cifar100": {
        "num_classes": 100,
        "in_channels": 3,
        "img_size": (32, 32),
        "mean": CIFAR100_MEAN,
        "std": CIFAR100_STD
    },
    "pcam": {
        "num_classes": 2,
        "in_channels": 3,
        "img_size": (64, 64),
        "mean": PCAM_MEAN,
        "std": PCAM_STD
    },
    "ham10000": {
        "num_classes": 7,
        "in_channels": 3,
        "img_size": (32, 32),
        "mean": HAM10000_MEAN,
        "std": HAM10000_STD
    }
}


def get_dataloaders(dataset_name="cifar10", data_dir="./data", batch_size=128, num_workers=None, seed=42):
    """
    Nạp DataLoaders chuẩn cho bài toán Split Learning.
    Trả về: (train_loader, test_loader, meta_dict)
    """
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Dataset '{dataset_name}' không được hỗ trợ. Chọn: {list(DATASET_METADATA.keys())}")

    meta = DATASET_METADATA[dataset_name].copy()
    if num_workers is None:
        num_workers = 0 if sys.platform == "win32" else 2

    if dataset_name == "cifar10":
        train_loader, test_loader = get_cifar10(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == "cifar100":
        train_loader, test_loader = get_cifar100(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == "pcam":
        # Ưu tiên nạp mảng NumPy đã tiền xử lý 64x64 nếu có
        candidates = [
            os.path.join(data_dir, "pcam_processed"),
            os.path.join(data_dir, "pcam"),
            data_dir
        ]
        processed_dir = None
        for c in candidates:
            if os.path.isfile(os.path.join(c, "train_images.npy")):
                processed_dir = c
                break

        if processed_dir is not None:
            train_loader, val_loader, test_loader, pcam_meta = get_pcam_dataloaders(
                data_dir=processed_dir, batch_size=batch_size, num_workers=num_workers
            )
            meta.update(pcam_meta)
            meta["val_loader"] = val_loader
        else:
            p_dir = os.path.join(data_dir, "pcam")
            train_loader, test_loader = get_pcam(data_dir=p_dir, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == "ham10000":
        h_dir = os.path.join(data_dir, "ham10000")
        train_loader, test_loader = get_ham10000(data_dir=h_dir, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader, meta


def get_fsha_dataloaders(dataset_name="cifar10", data_dir="./data", batch_size=128, priv_ratio=0.5, seed=42):
    """
    Nạp phân chia Private / Public cho tấn công chủ động FSHA.
    Trả về: (priv_loader, pub_loader, test_eval_loader, test_full_loader, meta_dict)
    """
    dataset_name = dataset_name.lower().strip()
    meta = DATASET_METADATA.get(dataset_name, DATASET_METADATA["cifar10"]).copy()
    num_workers = 0 if sys.platform == "win32" else 2

    if dataset_name == "cifar10":
        priv_loader, pub_loader, test_full_loader = get_cifar10_fsha_splits(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            priv_ratio=priv_ratio,
            seed=seed
        )
    else:
        train_loader, test_full_loader, _ = get_dataloaders(
            dataset_name=dataset_name, data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, seed=seed
        )
        if train_loader is None:
            return None, None, None, None, meta

        trainset = train_loader.dataset
        n_total = len(trainset)
        n_priv = int(n_total * priv_ratio)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=g).tolist()
        priv_indices = perm[:n_priv]
        pub_indices = perm[n_priv:]

        priv_set = Subset(trainset, priv_indices)
        pub_set = Subset(trainset, pub_indices)

        use_cuda = torch.cuda.is_available()
        priv_loader = DataLoader(priv_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
        pub_loader = DataLoader(pub_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)

    # Tập test_eval_loader đánh giá nhanh (2000 mẫu ngẫu nhiên cố định)
    g_eval = torch.Generator().manual_seed(seed)
    n_test = len(test_full_loader.dataset)
    eval_n = min(2000, n_test)
    eval_indices = torch.randperm(n_test, generator=g_eval)[:eval_n].tolist()
    test_eval_set = Subset(test_full_loader.dataset, eval_indices)
    use_cuda = torch.cuda.is_available()
    test_eval_loader = DataLoader(test_eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)

    return priv_loader, pub_loader, test_eval_loader, test_full_loader, meta
