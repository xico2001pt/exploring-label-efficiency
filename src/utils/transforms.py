import torch
import numpy as np


def temperature_sharpening(logits, temperature):
    logits = logits.pow(1 / temperature)
    return logits / logits.sum(dim=-1, keepdim=True)


def mixup(x1, x2, y1, y2, lam):
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y


class GaussianNoise(torch.nn.Module):
    """Add gaussian noise to the image.
    """
    def forward(self, *inputs):
        img = inputs[0]
        noise = torch.randn_like(img) * 0.15
        if len(inputs) == 1:
            return img + noise
        return img + noise, *inputs[1:]
