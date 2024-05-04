import torch
import torch.nn as nn


class LinearEvaluator(nn.Module):
    def __init__(self, backbone, num_classes):
        super(LinearEvaluator, self).__init__()
        self.num_classes = num_classes
        if backbone is not None:
            self.set_backbone(backbone)

    def set_backbone(self, backbone):
        self.backbone = backbone
        out_features = self.get_out_features(backbone)
        self.classifier = nn.Linear(out_features, self.num_classes)

    def get_out_features(self, backbone):
        out = backbone(torch.randn(1, 3, 32, 32))
        return out.size(1)

    def forward(self, x):
        x = self.backbone(x)
        if len(x.shape) > 2:
            x = x.mean([2, 3])
        x = self.classifier(x)
        return x
