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
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss = nn.CrossEntropyLoss(reduction="sum")
        self.mask = None

    def _create_mask(self, N, device):
        mask = torch.ones((2 * N, 2 * N), dtype=bool).to(device)
        mask = mask.fill_diagonal_(0)
        for i in range(N):
            mask[i, N + i] = 0
            mask[N + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)

        if self.mask is None:
            self.mask = self._create_mask(N, z.device)

        sim = self.sim(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, -N)

        positive = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * N, 1).to(z.device)
        negative = sim[self.mask].reshape(2 * N, -1)

        labels = torch.zeros(2 * N).to(z.device).long()
        logits = torch.cat((positive, negative), dim=1)
        loss = self.loss(logits, labels) / (2 * N)
        return {"total": loss} if self.return_dict else loss
