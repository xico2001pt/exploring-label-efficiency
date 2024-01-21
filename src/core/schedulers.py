from torch.optim.lr_scheduler import LambdaLR
from ..utils.ramps import exp_warmup


class ExpWarmupLR(LambdaLR):
    def __init__(self, optimizer, rampup_length, rampdown_length, num_epochs):
        lr_lambda = exp_warmup(rampup_length, rampdown_length, num_epochs)
        super().__init__(optimizer, lr_lambda)
