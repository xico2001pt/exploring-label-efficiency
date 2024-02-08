import torch
import numpy as np
from .pi_model import PiModel
from ...utils.structures import EMA


class TemporalEnsembling(PiModel):
    def __init__(self, w_max, unsupervised_weight_rampup_length, accumlation_decay):
        super().__init__(w_max, unsupervised_weight_rampup_length)
        self.accumlation_decay = accumlation_decay

    def on_start_train(self, train_data):
        super().on_start_train(train_data)

        total_size = train_data.dataset_size["total"]
        self.ema = EMA(self.accumlation_decay, np.zeros((total_size, self.num_classes)))

    def on_change_epoch(self, epoch):
        super().on_change_epoch(epoch)

        self.ensemble_predictions = self.ema.get_value() / (1 - self.accumlation_decay ** epoch)

    def get_predictions(self, idx, labeled, unlabeled):
        l1 = self.stochastic_augmentation(labeled)
        u1 = self.stochastic_augmentation(unlabeled)

        branch1 = torch.cat([l1, u1])

        self.pred1 = self.model(branch1)

        return self.pred1, self.pred1#self.ensemble_predictions[idx * self.pred1.size(0):(idx + 1) * self.pred1.size(0)]

    def update_ensemble_predictions(self, idx):
        index = idx * self.pred1.size(0)
        self.ema.update_partial(self.pred1, index)

    def compute_loss(self, idx, labeled, targets, unlabeled):
        res = super().compute_loss(idx, labeled, targets, unlabeled)

        self.update_ensemble_predictions(idx)

        return res
