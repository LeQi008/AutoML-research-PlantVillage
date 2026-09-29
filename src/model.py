import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ShuffleNet_V2_X0_5_Weights,
    SqueezeNet1_0_Weights,
    mobilenet_v3_small,
    resnet18,
    shufflenet_v2_x0_5,
    squeezenet1_0,
)


def count_parameters(model):
    """Return parameter counts using one consistent definition for every model."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def create_model(num_classes, pretrained=True, model_name="resnet18_v1"):
    """Create a supported image-classification model.

    ``model_name`` defaults to the existing ResNet18 implementation so callers
    using the original ``create_model(num_classes, pretrained)`` interface keep
    the same behavior.  The classifier replacement is deliberately kept here,
    rather than in the training pipeline, so every torchvision backbone emits
    exactly ``num_classes`` logits.
    """
    if model_name == "resnet18_v1":
        weights = ResNet18_Weights.DEFAULT if pretrained else None

        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None

        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )
        return model

    if model_name == "shufflenet_v2_x0_5":
        weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None

        model = shufflenet_v2_x0_5(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "squeezenet1_0":
        weights = SqueezeNet1_0_Weights.DEFAULT if pretrained else None

        model = squeezenet1_0(weights=weights)
        # SqueezeNet's classifier is a 1x1 convolution, not a Linear layer.
        # Keep its dropout and adaptive pooling while replacing only its output
        # channel count for the dataset's class count.
        classifier_conv = model.classifier[1]
        model.classifier[1] = nn.Conv2d(
            classifier_conv.in_channels,
            num_classes,
            kernel_size=classifier_conv.kernel_size,
            stride=classifier_conv.stride,
        )
        return model

    raise ValueError(f"Unknown model: {model_name}")
