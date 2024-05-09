import torch.nn as nn


class LinearEvaluator(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LinearEvaluator, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.fc(x)
