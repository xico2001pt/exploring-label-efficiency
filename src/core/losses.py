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

    def forward(self, proj1, proj2):
        proj = torch.stack([proj1, proj2], dim=1).view(2 * proj1.size(0), proj1.size(1))

        sim = F.cosine_similarity(proj[None, :, :], proj[:, None, :], dim=-1)
        sim[torch.eye(sim.size(0), dtype=torch.bool)] = float('-inf')
        sim /= self.temperature

        targets = torch.arange(2 * proj1.size(0)).to(proj.device)
        targets[0::2] += 1
        targets[1::2] -= 1

        loss = self.loss(sim, targets)

        self.labels = targets
        self.sim = sim

        return {"total": loss} if self.return_dict else loss

    def get_labels(self):
        return self.labels

    def get_sim(self):
        return self.sim
