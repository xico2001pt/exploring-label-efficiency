from .trainer import Trainer
from tqdm import tqdm


class SemiLTrainer(Trainer):
    def __init__(self, model, method, device, logger):
        super().__init__(model, device, logger)
        self.method = method

    def _compute_semi_supervised_loss(self, labeled, targets, unlabeled):
        labeled_outputs, loss = self.method(labeled, targets, unlabeled)
        return labeled_outputs, loss

    def _batch_iteration(self, dataloader, is_train, optimizer, metrics, total_loss, total_metrics, description):
        if is_train:
            labeled_dataloader, unlabeled_dataloader = dataloader

            # ...
        else:
            super()._batch_iteration(dataloader, is_train, optimizer, metrics, total_loss, total_metrics, description)
