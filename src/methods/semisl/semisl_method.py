class SemiSLMethod:
    def on_start_train(self, train_data):
        pass

    def on_change_epoch(self, epoch):
        pass

    def set_model(self, model):
        self.model = model

    def compute_loss(self, labeled, targets, unlabeled):
        raise NotImplementedError
