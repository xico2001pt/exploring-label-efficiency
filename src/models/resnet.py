import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.classifier = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.classifier(feat)
        return feat
