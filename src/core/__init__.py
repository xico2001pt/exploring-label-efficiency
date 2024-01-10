from .losses import CrossEntropyLoss
from .stop_conditions import StopPatience

classes = {
    "losses": [CrossEntropyLoss],
    "stop_conditions": [StopPatience]
}
