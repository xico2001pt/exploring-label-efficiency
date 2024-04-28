from ..method import Method


class SemiSLMethod(Method):
    def compute_loss(self, idx, labeled, targets, unlabeled):
        raise NotImplementedError
