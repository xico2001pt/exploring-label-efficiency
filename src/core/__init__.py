from .losses import CrossEntropyLoss
from torchmetrics import Accuracy, Dice, JaccardIndex
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, PolynomialLR, CosineAnnealingLR
from .schedulers import ExpWarmupLR
from .stop_conditions import StopPatience

classes = {
    "losses": [CrossEntropyLoss],  # Add the loss classes here
    "metrics": [Accuracy, Dice, JaccardIndex],  # Add the metric classes here
    "optimizers": [Adam, SGD],  # Add the optimizer classes here
    "schedulers": [
        ExponentialLR,
        ExpWarmupLR,
        MultiStepLR,
        PolynomialLR,
        CosineAnnealingLR
    ],  # Add the scheduler classes here
    "stop_conditions": [StopPatience]  # Add the stop condition classes here
}
