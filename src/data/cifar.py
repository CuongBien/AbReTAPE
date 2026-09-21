# Module tải và nạp dữ liệu CIFAR-10 / CIFAR-100
import sys
import torch
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

DEFAULT_NUM_WORKERS = 0 if sys.platform == "win32" else 2


def get_cifar10(data_dir="./data", batch_size=128, num_workers=None):
    """
    Nạp dữ liệu CIFAR-10 với chuẩn hóa và data augmentation chuẩn:
    - Train: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize
    - Test: Normalize
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS
    return get_cifar(dataset="cifar10", data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)


def get_cifar100(data_dir="./data", batch_size=128, num_workers=None):
    """
    Nạp dữ liệu CIFAR-100 cho tầng kiểm chứng controlled verification.
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS
    return get_cifar(dataset="cifar100", data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)


def get_cifar(dataset="cifar10", data_dir="./data", batch_size=128, num_workers=None):
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS
    is_cifar10 = dataset.lower() == "cifar10"
    mean = CIFAR10_MEAN if is_cifar10 else CIFAR100_MEAN
    std = CIFAR10_STD if is_cifar10 else CIFAR100_STD
    dataset_cls = torchvision.datasets.CIFAR10 if is_cifar10 else torchvision.datasets.CIFAR100

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    trainset = dataset_cls(root=data_dir, train=True, download=True, transform=train_tf)
    testset = dataset_cls(root=data_dir, train=False, download=True, transform=test_tf)

    use_cuda = torch.cuda.is_available()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda
    )
    return trainloader, testloader
