# -*- coding: utf-8 -*-
# Kiến trúc ResNet-18 điều chỉnh và mô hình Client/Server cho Split Learning
import torch
import torch.nn as nn
import torchvision.models as models


def resnet18_cifar(num_classes=10):
    """
    ResNet-18 điều chỉnh cho CIFAR-10 / CIFAR-100 (ảnh 32x32):
    - conv1: 3x3, stride 1, padding 1 thay cho 7x7 stride 2
    - maxpool: Identity thay vì downsampling
    """
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # bỏ maxpool vì ảnh nhỏ
    model.fc = nn.Linear(512, num_classes)
    return model


class ClientModel(nn.Module):
    """
    Phần Client: stem (conv1 + bn1 + relu) + layer1.
    - Với CIFAR-10/100 (ảnh 32x32): stem_stride=1 -> z shape (B, 64, 32, 32).
    - Với PCAM (ảnh 64x64): stem_stride=2 -> z shape (B, 64, 32, 32).
    Cả hai bài toán đều cho ra biểu diễn smashed data z có cùng kích thước:
    (B, 64, 32, 32) với tổng số chiều D = 64 * 32 * 32 = 65,536!
    """
    def __init__(self, in_channels: int = 3, stem_stride: int = 1):
        super().__init__()
        base = models.resnet18(weights=None)
        conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=stem_stride, padding=1, bias=False)
        self.stem = nn.Sequential(conv1, base.bn1, base.relu)
        self.layer1 = base.layer1
        self.stem_stride = stem_stride
        self.in_channels = in_channels

    def forward(self, x):
        return self.layer1(self.stem(x))


class ServerModel(nn.Module):
    """
    Phần Server: layer2 + layer3 + layer4 + avgpool + fc.
    Nhận biểu diễn trung gian z (B, 64, 32, 32) và tính logits phân loại.
    Dùng chung không đổi cho cả CIFAR (num_classes=10) và PCAM (num_classes=2).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18_cifar(num_classes)
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc
        self.num_classes = num_classes

    def forward(self, z):
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        return self.fc(z)
