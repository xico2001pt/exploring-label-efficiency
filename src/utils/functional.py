import torch


def temperature_sharpening(logits, temperature):
    logits = logits.pow(1 / temperature)
    return logits / logits.sum(dim=-1, keepdim=True)


def mixup(x1, x2, y1, y2, lam):
    x = torch.add(lam * x1, (1 - lam) * x2)
    y = torch.add(lam * y1, (1 - lam) * y2)
    return x, y
