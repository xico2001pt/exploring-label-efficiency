class SemiSLMethod:
    def on_change_epoch(self, epoch):
        pass

    def set_model(self, model):
        self.model = model

    def truncate_batches(self):
        raise NotImplementedError

    def compute_loss(self, labeled, targets, unlabeled):
        raise NotImplementedError
