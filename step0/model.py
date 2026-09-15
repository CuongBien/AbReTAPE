# Bước 0 — Vanilla Split Learning: mô hình và điểm cắt
import torch
import torch.nn as nn
import torchvision.models as models


def resnet18_cifar(num_classes=10):
    # ResNet-18 điều chỉnh cho CIFAR-10 (ảnh 32x32)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # bỏ maxpool vì ảnh nhỏ
    model.fc = nn.Linear(512, num_classes)
    return model


class ClientModel(nn.Module):
    # Phần client: stem + layer1. Đầu ra là smashed data z (IR)
    def __init__(self):
        super().__init__()
        base = resnet18_cifar()
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.layer1 = base.layer1

    def forward(self, x):
        return self.layer1(self.stem(x))


class ServerModel(nn.Module):
    # Phần server: layer2..4 + avgpool + fc
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18_cifar(num_classes)
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, z):
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        return self.fc(z)
