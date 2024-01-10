from .losses import CrossEntropyLoss
from torchmetrics import Accuracy
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from .stop_conditions import StopPatience

classes = {
    "losses": [CrossEntropyLoss],  # Add the loss classes here
    "metrics": [Accuracy],  # Add the metric classes here
    "optimizers": [Adam, SGD],  # Add the optimizer classes here
    "schedulers": [ExponentialLR, MultiStepLR],  # Add the scheduler classes here
    "stop_conditions": [StopPatience]  # Add the stop condition classes here
}
