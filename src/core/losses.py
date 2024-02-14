import torch.nn as nn
import torchmetrics


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        loss = super().forward(input, target)
        return {"total": loss}


# TODO: Not working
class DiceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DiceLoss, self).__init__()
        self.dice = torchmetrics.Dice(*args, **kwargs)

    def forward(self, input, target):
        print(input.device, target.device)
        loss = 1 - self.dice.to(input.device)(input, target)
        return {"total": loss}
