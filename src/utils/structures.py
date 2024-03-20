import torch
from .utils import classes_mean


class EMA:  # Exponential Moving Average
    def __init__(self, decay, initial):
        self.decay = decay
        self.value = initial

    def update(self, value):
        self.value = self.decay * self.value + (1 - self.decay) * value

    def update_partial(self, value, start_index, size):
        self.value[start_index:start_index + size] = self.decay * self.value[start_index:start_index + size] + (1 - self.decay) * value

    def get_value(self):
        return self.value


class TensorMovingAverage:
    def __init__(self, window_size, num_classes, device):
        self.window_size = window_size
        self.values = torch.ones((window_size, num_classes)).to(device) / num_classes

    def update(self, value):
        value = classes_mean(value)
        self.values = torch.cat([self.values[1:], value.unsqueeze(0)], dim=0)

    def get_value(self):
        return self.values.mean(dim=0)
