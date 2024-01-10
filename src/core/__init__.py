from .losses import CrossEntropyLoss
from torchmetrics import Accuracy
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from .stop_conditions import StopPatience

classes = {
    "losses": [CrossEntropyLoss],
    "metrics": [Accuracy],
    "optimizers": [Adam, SGD],
    "schedulers": [StepLR],
    "stop_conditions": [StopPatience]
}
