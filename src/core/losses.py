import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        loss = super().forward(input, target)
        return {"total": loss}


class CrossEntropyWithLogitsLoss(nn.Module):
    def __init__(self, return_dict=True):
        super(CrossEntropyWithLogitsLoss, self).__init__()
        self.return_dict = return_dict
        self.loss = F.log_softmax

    def forward(self, input, target):
        loss = self.loss(input, dim=1)
        loss = torch.sum(loss * target, dim=1)
        loss = -torch.mean(loss)
        return {"total": loss} if self.return_dict else loss
