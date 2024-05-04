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


class NTXent9Loss(nn.Module):
    def __init__(self, temperature=0.5, return_dict=True):
        super(NTXent9Loss, self).__init__()
        self.temperature = temperature
        self.return_dict = return_dict
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.cossim = nn.CosineSimilarity(dim=2)
        self.mask = None

    def _create_mask(self, batch_size, device):
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, proj1, proj2):
        batch_size = proj1.shape[0]

        if self.mask is None:
            self.mask = self._create_mask(batch_size, proj1.device)

        projs = torch.cat([proj1, proj2], dim=0)

        sim = self.cossim(projs.unsqueeze(1), projs.unsqueeze(0)) / self.temperature

        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)

        positive = torch.cat((sim_ij, sim_ji), dim=0).reshape(2 * batch_size, 1)
        negative = sim[self.mask].reshape(2 * batch_size, -1)

        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=proj1.device)
        logits = torch.cat((positive, negative), dim=1)

        loss = self.loss(logits, labels) / (2 * batch_size)

        self.labels = labels
        self.sim = logits

        return {"total": loss} if self.return_dict else loss

    def get_labels(self):
        return self.labels

    def get_sim(self):
        return self.sim


class NTXentFinalLoss(nn.Module):
    def __init__(self, temperature=0.5, return_dict=True):
        super(NTXentFinalLoss, self).__init__()
        self.temperature = temperature
        self.return_dict = return_dict
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, proj1, proj2):
        projs = torch.cat([proj1, proj2], dim=0)
        n_samples = projs.shape[0]

        # Full Similarity Matrix
        sim = torch.mm(projs, projs.t().contiguous())
        sim = torch.exp(sim / self.temperature)

        # Negative Similarity
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive Similarity
        pos = torch.exp(torch.sum(proj1 * proj2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # Loss
        loss = -torch.log(pos / neg).mean()

        self.sim = sim
        self.labels = torch.arange(n_samples).to(sim.device)
        self.labels[::2] += 1
        self.labels[1::2] -= 1

        return {"total": loss} if self.return_dict else loss
    
    def get_labels(self):
        return self.labels
    
    def get_sim(self):
        return self.sim



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


class NTXent5Loss(nn.Module):
    def __init__(self, temperature=0.5, return_dict=True):
        super(NTXent5Loss, self).__init__()
        self.temperature = temperature
        self.return_dict = return_dict
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, proj1, proj2):
        batch_size = proj1.size(0)

        masks = torch.eye(batch_size, batch_size, dtype=torch.bool).to(proj1.device)

        logits_aa = torch.matmul(proj1, proj1.t()) / self.temperature
        logits_aa.masked_fill_(masks, value=torch.finfo(logits_aa.dtype).min)

        logits_bb = torch.matmul(proj2, proj2.t()) / self.temperature
        logits_bb.masked_fill_(masks, value=torch.finfo(logits_bb.dtype).min)

        logits_ab = torch.matmul(proj1, proj2.t()) / self.temperature
        logits_ba = torch.matmul(proj2, proj1.t()) / self.temperature

        targets = torch.arange(batch_size).to(proj1.device).long()
        loss_a = self.loss(torch.cat([logits_ab, logits_aa], dim=-1), targets)
        loss_b = self.loss(torch.cat([logits_ba, logits_bb], dim=-1), targets)
        
        loss = (loss_a + loss_b) / 2

        self.labels = torch.cat([targets, targets])
        self.sim = torch.cat([torch.cat([logits_ab, logits_aa], dim=-1), torch.cat([logits_ba, logits_bb], dim=-1)], dim=0)

        return {"total": loss} if self.return_dict else loss

    def get_labels(self):
        return self.labels
    
    def get_sim(self):
        return self.sim


class NTXent3Loss(nn.Module):
    def __init__(self, temperature=0.5, return_dict=True):
        super(NTXent3Loss, self).__init__()
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


class NTXent4Loss(nn.Module):
    def __init__(self, temperature=0.5, return_dict=True):
        super(NTXent4Loss, self).__init__()
        self.temperature = temperature
        self.return_dict = return_dict
        self.loss = nn.CrossEntropyLoss()

    def forward(self, proj1, proj2):
        feat = torch.stack([proj1, proj2], dim=1).view(2 * proj1.size(0), proj1.size(1))
        feat = F.normalize(feat, dim=1)
        feat_scores = (feat @ feat.t()).clamp(min=1e-7)
        feat_scores /= self.temperature
        feat_scores = feat_scores - torch.eye(feat_scores.size(0)).to(feat_scores.device) * 1e5

        labels = torch.arange(feat.size(0)).to(feat.device)
        labels[::2] += 1
        labels[1::2] -= 1

        loss = self.loss(feat_scores, labels.long())

        self.labels = labels
        self.sim = feat_scores
        return {"total": loss} if self.return_dict else loss

    def get_labels(self):
        return self.labels

    def get_sim(self):
        return self.sim

if __name__ == '__main__':
    import torch
    import torch.nn as nn

    proj1 = torch.randn(4, 3)
    proj2 = torch.randn(4, 3)

    loss = NTXentLoss(temperature=0.5)
    loss2 = NTXent2Loss(temperature=0.5)
    loss3 = NTXent3Loss(temperature=0.5)
    loss4 = NTXent4Loss(temperature=0.5)
    loss5 = NTXent5Loss(temperature=0.5)
    lossReal = NTXentFinalLoss(temperature=0.5)
    loss9 = NTXent9Loss(temperature=0.5)

    print(loss(proj1, proj2)['total'])
    print(loss2(proj1, proj2)['total'])
    print(loss3(proj1, proj2)['total'])
    print(loss4(proj1, proj2)['total'])
    print(loss5(proj1, proj2)['total'])
    print(lossReal(proj1, proj2)['total'])
    print(loss9(proj1, proj2)['total'])