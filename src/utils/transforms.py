import torch


def temperature_sharpening(logits, temperature):
    logits = logits.pow(1 / temperature)
    return logits / logits.sum(dim=-1, keepdim=True)


def mixup(x1, x2, y1, y2, lam):
    x = lam.view(-1, 1, 1, 1) * x1 + (1 - lam.view(-1, 1, 1, 1)) * x2
    y = lam.view(-1, 1) * y1 + (1 - lam.view(-1, 1)) * y2  # For classification
    #y = lam.view(-1, 1, 1, 1) * y1 + (1 - lam.view(-1, 1, 1, 1)) * y2  # For segmentation
    # TODO: Should adapt to the shape of y
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
