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
        self.cossim = nn.CosineSimilarity(dim=-1)

    def forward(self, proj1, proj2):
        batch_size = proj1.size(0)

        sim11 = self.cossim(proj1.unsqueeze(-2), proj1.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(proj2.unsqueeze(-2), proj2.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(proj1.unsqueeze(-2), proj2.unsqueeze(-3)) / self.temperature

        d = sim12.size(-1)
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')

        sim1 = torch.cat([sim12, sim11], dim=-1)
        sim2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)

        self.sim = torch.cat([sim1, sim2], dim=-2)

        self.labels = torch.arange(2 * batch_size).to(proj1.device)
        loss = self.loss(self.sim, self.labels)
        return {"total": loss} if self.return_dict else loss

    def get_labels(self):
        return self.labels

    def get_sim(self):
        return self.sim


class NTXent2Loss(nn.Module):
    def __init__(self, temperature=0.5, return_dict=True):
        super(NTXent2Loss, self).__init__()
        self.temperature = temperature
        self.return_dict = return_dict

    def forward(self, proj1, proj2):
        batch_size = proj1.size(0)

        proj1, proj2 = F.normalize(proj1, dim=-1), F.normalize(proj2, dim=-1)
        projections = torch.cat([proj1, proj1], dim=0)

        sim_matrix = torch.exp(torch.mm(projections, projections.t().contiguous()) / self.temperature)

        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        pos_sim = torch.exp(torch.sum(proj1 * proj2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1)).mean()
        return {"total": loss} if self.return_dict else loss


class NTXent3Loss(nn.Module):
    def __init__(self, temperature=0.5, return_dict=True):
        super(NTXent3Loss, self).__init__()
        self.temperature = temperature
        self.return_dict = return_dict
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feat):
        feat = F.normalize(feat, dim=-1)
        feat_scores = torch.matmul(feat, feat.t()).clamp(min=1e-7) / self.temperature
        feat_scores = feat_scores - torch.eye(feat_scores.size(0)).to(feat_scores.device) * 1e5

        labels = torch.arange(feat.size(0)).to(feat.device)
        labels[::2] += 1
        labels[1::2] -= 1

        loss = self.loss(feat_scores, labels.long())
        return {"total": loss} if self.return_dict else loss
