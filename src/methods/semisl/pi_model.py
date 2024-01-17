import numpy as np
from .semisl_method import SemiSLMethod


class PiModel(SemiSLMethod):
    def __init__(self, max_unsupervised_weight):
        pass

    def on_change_epoch(self, epoch):
        pass

    def truncate_batches(self):
        return False

    def compute_loss(self, labeled, targets, unlabeled):
        raise NotImplementedError()
