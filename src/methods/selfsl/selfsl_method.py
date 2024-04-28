from ..method import Method


class SelfSLMethod(Method):
    def compute_loss(self, idx, unlabeled):
        raise NotImplementedError
