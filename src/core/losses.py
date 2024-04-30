import torch
import torch.nn as nn
import torch.nn.functional as F


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


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, return_dict=True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.return_dict = return_dict
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feat):
        feat = F.normalize(feat, dim=1)
        feat_scores = torch.matmul(feat, feat.t()).clamp(min=1e-7) / self.temperature
        feat_scores = feat_scores - torch.eye(feat_scores.size(0)).to(feat_scores.device) * 1e5

        labels = torch.arange(feat.size(0)).to(feat.device)
        labels[::2] += 1
        labels[1::2] -= 1

        loss = self.loss(feat_scores, labels.long())
        return {"total": loss} if self.return_dict else loss
