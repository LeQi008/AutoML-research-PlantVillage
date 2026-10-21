import torch
import torch.nn as nn
from torchvision.models import (MobileNet_V3_Small_Weights, ResNet18_Weights, ShuffleNet_V2_X0_5_Weights, SqueezeNet1_0_Weights, mobilenet_v3_small, resnet18, shufflenet_v2_x0_5, squeezenet1_0)

class DepthwiseSeparableConv(nn.Module):
    """Depthwise convolution followed by pointwise convolution."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))
    def forward(self, x): return self.block(x)

class CustomDSCNNV1(nn.Module):
    """Existing compact depthwise-separable CNN; retained for compatibility."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True), DepthwiseSeparableConv(32, 32), DepthwiseSeparableConv(32, 64, 2), DepthwiseSeparableConv(64, 64), DepthwiseSeparableConv(64, 128, 2), DepthwiseSeparableConv(128, 128), DepthwiseSeparableConv(128, 256, 2), DepthwiseSeparableConv(256, 256), DepthwiseSeparableConv(256, 384, 2), DepthwiseSeparableConv(384, 384))
        self.pool, self.classifier = nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(384, num_classes)
    def forward(self, x): return self.classifier(self.pool(self.features(x)).flatten(1))

class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual with a linear bottleneck."""
    def __init__(self, in_channels, out_channels, stride=1, expansion=2):
        super().__init__()
        if stride not in (1, 2) or expansion < 1: raise ValueError("Invalid inverted-residual configuration")
        hidden = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(nn.Conv2d(in_channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True), nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True), nn.Conv2d(hidden, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
    def forward(self, x): return x + self.block(x) if self.use_residual else self.block(x)

class CustomIRCNNV1(nn.Module):
    """Existing sixteen-block model; retained for compatibility."""
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        config = [(24,2,2),(24,1,2),(32,2,2),(32,1,2),(48,2,2),(48,1,2),(80,2,3),(80,1,3),(80,1,3),(112,2,3),(112,1,2),(112,1,2),(160,2,3),(160,1,3),(160,1,3),(160,1,3)]
        blocks, channels = [], 32
        for out, stride, expansion in config:
            blocks.append(InvertedResidual(channels, out, stride, expansion)); channels = out
        self.features, self.pool, self.classifier = nn.Sequential(*blocks), nn.AdaptiveAvgPool2d((1,1)), nn.Linear(160, num_classes)
    def forward(self, x): return self.classifier(self.pool(self.features(self.stem(x))).flatten(1))

class ResidualConvBlock(nn.Module):
    """Conventional two-convolution residual block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False), nn.BatchNorm2d(out_channels)
        self.conv2, self.bn2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = self.shortcut(x); x = self.activation(self.bn1(self.conv1(x)))
        return self.activation(self.bn2(self.conv2(x)) + residual)

class SelectiveDepthwiseResidualBlock(nn.Module):
    """Depthwise-separable residual block used selectively in v2."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise, self.bn1 = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False), nn.BatchNorm2d(in_channels)
        self.pointwise, self.bn2 = nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
        self.activation = nn.ReLU6(inplace=True)
    def forward(self, x):
        residual = self.shortcut(x); x = self.activation(self.bn1(self.depthwise(x)))
        return self.activation(self.bn2(self.pointwise(x)) + residual)

class CustomIRCNNV2(nn.Module):
    """Shallow residual CNN with four downsampling transitions and 734,470 parameters for 38 classes."""
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.features = nn.Sequential(ResidualConvBlock(32, 64, 2), ResidualConvBlock(64, 64), SelectiveDepthwiseResidualBlock(64, 128, 2), SelectiveDepthwiseResidualBlock(128, 128), ResidualConvBlock(128, 128), SelectiveDepthwiseResidualBlock(128, 192, 2), SelectiveDepthwiseResidualBlock(192, 192), SelectiveDepthwiseResidualBlock(192, 256, 2), SelectiveDepthwiseResidualBlock(256, 256))
        self.pool, self.classifier = nn.AdaptiveAvgPool2d((1,1)), nn.Linear(256, num_classes)
    def forward(self, x): return self.classifier(self.pool(self.features(self.stem(x))).flatten(1))

class ECA(nn.Module):
    """Efficient channel attention with a small local 1-D channel interaction."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1); self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False); self.activation = nn.Sigmoid()
    def forward(self, x):
        weights = self.pool(x).squeeze(-1).transpose(-1, -2); weights = self.activation(self.conv(weights)).transpose(-1, -2).unsqueeze(-1)
        return x * weights

class UlenInvertedResidual(nn.Module):
    """MobileNetV2 inverted/depthwise block, optionally followed by late ECA."""
    def __init__(self, in_channels, out_channels, stride=1, expansion=3, attention=False):
        super().__init__(); self.block = InvertedResidual(in_channels, out_channels, stride, expansion); self.attention = ECA(out_channels) if attention else nn.Identity()
    def forward(self, x): return self.attention(self.block(x))

class CustomULENV1(nn.Module):
    """Compact ULEN/MobileNetV2-inspired model with a multi-scale pooling head."""
    def __init__(self, num_classes):
        super().__init__(); self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        config = [(32,32,1,2,False),(32,48,2,3,False),(48,48,1,3,False),(48,80,2,3,False),(80,80,1,3,False),(80,128,2,3,False),(128,128,1,3,True),(128,192,2,3,False),(192,192,1,3,True)]
        self.features = nn.Sequential(*[UlenInvertedResidual(*item) for item in config]); self.projection = nn.Sequential(nn.Conv2d(192, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU6(inplace=True)); self.global_pool = nn.AdaptiveAvgPool2d((1,1)); self.local_pool = nn.AdaptiveAvgPool2d((2,2)); self.classifier = nn.Linear(256*5, num_classes)
    def forward(self, x):
        x = self.projection(self.features(self.stem(x))); return self.classifier(torch.cat((self.global_pool(x).flatten(1), self.local_pool(x).flatten(1)), dim=1))

class CustomULENV2(nn.Module):
    """Lighter ULEN candidate with low-expansion residual stages and a compact SPP head."""
    def __init__(self, num_classes):
        super().__init__(); self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        config = [(32,32,1,2,False),(32,48,2,2,False),(48,48,1,2,False),(48,80,2,2,False),(80,80,1,2,False),(80,128,2,2,False),(128,128,1,2,False),(128,160,2,2,True)]
        self.features = nn.Sequential(*[UlenInvertedResidual(*item) for item in config]); self.projection = nn.Sequential(nn.Conv2d(160, 224, 1, bias=False), nn.BatchNorm2d(224), nn.ReLU6(inplace=True)); self.global_pool = nn.AdaptiveAvgPool2d((1,1)); self.pool_2 = nn.AdaptiveAvgPool2d((2,2)); self.pool_3 = nn.AdaptiveAvgPool2d((3,3)); self.classifier = nn.Linear(224*14, num_classes)
    def forward(self, x):
        x = self.projection(self.features(self.stem(x))); pooled = (self.global_pool(x).flatten(1), self.pool_2(x).flatten(1), self.pool_3(x).flatten(1)); return self.classifier(torch.cat(pooled, dim=1))

CUSTOM_MODEL_NAMES = {"custom_ds_cnn_v1", "custom_ir_cnn_v1", "custom_ir_cnn_v2", "custom_ulen_v1", "custom_ulen_v2"}
EXPECTED_PARAMETER_COUNTS = {"custom_ir_cnn_v2": 734470, "custom_ulen_v1": 675925, "custom_ulen_v2": 401673}

def count_parameters(model):
    """Return total and trainable parameter counts."""
    return (sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))

def create_model(num_classes, pretrained=True, model_name="resnet18_v1"):
    """Create a supported model; custom models are always from scratch."""
    if model_name in CUSTOM_MODEL_NAMES:
        # Ignore the caller's flag deliberately: custom architectures never load weights.
        pretrained = False
        if model_name == "custom_ds_cnn_v1": return CustomDSCNNV1(num_classes)
        if model_name == "custom_ir_cnn_v1": return CustomIRCNNV1(num_classes)
        if model_name == "custom_ir_cnn_v2": return CustomIRCNNV2(num_classes)
        if model_name == "custom_ulen_v1": return CustomULENV1(num_classes)
        return CustomULENV2(num_classes)
    weights = None
    if model_name == "resnet18_v1":
        weights = ResNet18_Weights.DEFAULT if pretrained else None; model = resnet18(weights=weights); model.fc = nn.Linear(model.fc.in_features, num_classes); return model
    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None; model = mobilenet_v3_small(weights=weights); model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes); return model
    if model_name == "mobilenet_v3_small_linear":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None; model = mobilenet_v3_small(weights=weights); model.classifier = nn.Linear(model.classifier[0].in_features, num_classes); return model
    if model_name == "shufflenet_v2_x0_5":
        weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None; model = shufflenet_v2_x0_5(weights=weights); model.fc = nn.Linear(model.fc.in_features, num_classes); return model
    if model_name == "squeezenet1_0":
        weights = SqueezeNet1_0_Weights.DEFAULT if pretrained else None; model = squeezenet1_0(weights=weights); c = model.classifier[1]; model.classifier[1] = nn.Conv2d(c.in_channels, num_classes, kernel_size=c.kernel_size, stride=c.stride); return model
    raise ValueError(f"Unknown model: {model_name}")

