import torch
import torch.nn as nn
from torchvision.models import resnet50

class ModifiedResNet50(nn.Module):
    """
    Customized ResNet50 backbone designed for forensic detection.
    Modifications include disabling max-pooling and adjusting strides to
    preserve spatial resolution in the early layers.
    """
    def __init__(self, pretrain=True):
        super().__init__()
        self.backbone = resnet50(pretrained=pretrain)
        self.backbone.maxpool = nn.Identity()
        self.backbone.layer1[0].conv1.stride = (1, 1)
        self.backbone.layer1[0].downsample[0].stride = (1, 1)
        self.backbone.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.backbone(x)

class SourceSpecificProbing(nn.Module):
    """
    High-level module wrapper for source attribution tasks.
    """
    def __init__(self, pretrain=True):
        super().__init__()
        self.disc = ModifiedResNet50(pretrain=pretrain)

    def forward(self, x):
        return self.disc(x)

class FeatureExtractor(nn.Module):
    """
    Extraction module to retrieve deep latent representations
    from the Global Average Pooling layer.
    """
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(
            original_model.disc.backbone.conv1,
            original_model.disc.backbone.bn1,
            original_model.disc.backbone.relu,
            original_model.disc.backbone.maxpool,
            original_model.disc.backbone.layer1,
            original_model.disc.backbone.layer2,
            original_model.disc.backbone.layer3,
            original_model.disc.backbone.layer4,
            original_model.disc.backbone.avgpool
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x