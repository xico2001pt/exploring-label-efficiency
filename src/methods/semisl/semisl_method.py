class SemiSLMethod:
    def set_model(self, model):
        self.model = model

    def on_start_train(self, train_data):
        pass

    def on_start_epoch(self, epoch):
        pass

    def on_end_train(self, train_data):
        pass

    def on_end_epoch(self, epoch):
        pass

    def compute_loss(self, idx, labeled, targets, unlabeled):
        raise NotImplementedError
