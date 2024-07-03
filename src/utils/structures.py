import torch
from collections import OrderedDict
from .utils import classes_mean


class EMA:  # Exponential Moving Average
    def __init__(self, decay, initial):
        self.decay = decay
        self.value = initial

    def update(self, value):
        self.value = self.decay * self.value + (1 - self.decay) * value

    def update_partial(self, value, start_index, size):
        self.value[start_index:start_index + size] = (
            self.decay * self.value[start_index:start_index + size] +
            (1 - self.decay) * value
        )

    def get_value(self):
        return self.value


class EMAv2:  # Exponential Moving Average v2
    def __init__(self, beta):
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return self.beta * old + (1 - self.beta) * new

    def update_model(self, ema_model, model):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data = self.update_average(ema_param.data, param.data)


class BackboneWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        output = self.backbone(x)

        if isinstance(output, OrderedDict):
            output = output['out']

        if len(output.shape) > 2:
            output = output.mean([2, 3])

        return output


class TensorMovingAverage:
    def __init__(self, window_size, num_classes, device):
        self.window_size = window_size
        self.values = torch.ones((window_size, num_classes)).to(device) / num_classes

    def update(self, value):
        value = classes_mean(value)
        self.values = torch.cat([self.values[1:], value.unsqueeze(0)], dim=0)

    def get_value(self):
        return self.values.mean(dim=0)


def MultiLayerPerceptron(input_size, hidden_size, output_size):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.BatchNorm1d(hidden_size),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(hidden_size, output_size)
    )
