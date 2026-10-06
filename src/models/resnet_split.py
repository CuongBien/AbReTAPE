# Module tương thích (alias) cho src.models.resnet
from .resnet import ClientModel, ServerModel, resnet18_cifar

__all__ = ["ClientModel", "ServerModel", "resnet18_cifar"]
