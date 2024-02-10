import torch.nn as nn
import torchmetrics


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        loss = super().forward(input, target)
        return {"total": loss}


class DiceLoss(torchmetrics.Dice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        loss = super().forward(input, target)
        return {"total": 1 - loss}
