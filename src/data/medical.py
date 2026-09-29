# Module nạp dữ liệu Y tế cho Tầng 2 (Application Case Study): HAM10000 và PCAM
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image

DEFAULT_NUM_WORKERS = 0 if sys.platform == "win32" else 2

# Thông số chuẩn hóa cho ảnh mô bệnh học (Histopathology / PCAM) và da liễu (Dermatology / HAM10000)
PCAM_MEAN = (0.7008, 0.5384, 0.6916)
PCAM_STD = (0.2350, 0.2774, 0.2129)

HAM10000_MEAN = (0.7630, 0.5456, 0.5700)
HAM10000_STD = (0.1409, 0.1526, 0.1699)


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
            # Tìm thư mục ảnh
            img_dirs = [
                os.path.join(data_dir, "HAM10000_images_part_1"),
                os.path.join(data_dir, "HAM10000_images_part_2"),
                os.path.join(data_dir, "images"),
                data_dir
            ]

            # Phân chia train/test đơn giản (80/20) nếu không có cột split
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
    Nạp dữ liệu PatchCamelyon (PCAM) cho bài toán nhị phân y tế.
    Tự động tải qua torchvision.datasets.PCAM nếu chưa có.
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
