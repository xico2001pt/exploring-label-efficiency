# Code source:
# https://github.com/iBelieveCJM/meanteacher-pytorch/blob/master/util/ramps.py

import numpy as np


def linear_rampup(rampup_length):
    def warpper(epoch):
        return np.clip(epoch / rampup_length, 0.0, 1.0)
    return warpper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper


def exp_rampdown(rampdown_length, num_epochs):
    """Exponential rampdown from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = .5 * (epoch - (num_epochs - rampdown_length))
            return float(np.exp(-(ep * ep) / rampdown_length))
        else:
            return 1.0
    return warpper


def exp_warmup(rampup_length, rampdown_length, num_epochs):
    rampup = exp_rampup(rampup_length)
    rampdown = exp_rampdown(rampdown_length, num_epochs)

    def warpper(epoch):
        return rampup(epoch)*rampdown(epoch)
    return warpper
