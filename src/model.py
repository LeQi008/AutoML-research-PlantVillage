import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_model(num_classes, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model