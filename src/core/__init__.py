from .losses import CrossEntropyLoss, DiceLoss
from torchmetrics import Accuracy, Dice, JaccardIndex
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, PolynomialLR
from .schedulers import ExpWarmupLR
from .stop_conditions import StopPatience

classes = {
    "losses": [CrossEntropyLoss, DiceLoss],  # Add the loss classes here
    "metrics": [Accuracy, Dice, JaccardIndex],  # Add the metric classes here
    "optimizers": [Adam, SGD],  # Add the optimizer classes here
    "schedulers": [ExponentialLR, ExpWarmupLR, MultiStepLR, PolynomialLR],  # Add the scheduler classes here
    "stop_conditions": [StopPatience]  # Add the stop condition classes here
}
