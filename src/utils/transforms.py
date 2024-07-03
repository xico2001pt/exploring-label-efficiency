import torch
import torchvision.transforms.v2 as v2


def temperature_sharpening(logits, temperature):
    logits = logits.pow(1 / temperature)
    return logits / logits.sum(dim=1, keepdim=True)


def mixup(x1, x2, y1, y2, lam):
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y


class GaussianNoise(torch.nn.Module):
    """Add gaussian noise to the image.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, *inputs):
        if torch.rand(1) > self.p:
            return inputs[0] if len(inputs) == 1 else inputs

        img = inputs[0]
        noise = torch.randn_like(img) * 0.15
        if len(inputs) == 1:
            return img + noise
        return img + noise, *inputs[1:]


class InvariantRandAugment(v2.RandAugment):
    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),
        "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (
            lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4))).round().int(),
            False,
        ),
        "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }
