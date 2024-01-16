class PiModel:
    def __init__(self, max_unsupervised_weight):
        pass

    def truncate_batches(self):
        return False

    def compute_loss(self, labeled, targets, unlabeled):
        raise NotImplementedError()
