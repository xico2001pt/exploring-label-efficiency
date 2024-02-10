import torch
from .pi_model import PiModel
from ...utils.structures import EMA


class TemporalEnsembling(PiModel):
    def __init__(self, w_max, unsupervised_weight_rampup_length, accumlation_decay):
        super().__init__(w_max, unsupervised_weight_rampup_length)
        self.accumlation_decay = accumlation_decay

    def on_start_train(self, train_data):
        super().on_start_train(train_data)

        total_size = train_data.dataset_size["total"]
        self.ema = EMA(self.accumlation_decay, torch.zeros(size=(total_size, self.num_classes), requires_grad=False, device=train_data.device, dtype=torch.float32))
        self.ensemble_predictions = self.ema.get_value().detach().clone()

    def on_end_epoch(self, epoch):
        super().on_end_epoch(epoch)

        with torch.no_grad():
            self.ensemble_predictions = self.ema.get_value().detach().clone() / (1 - self.accumlation_decay ** epoch)

    def get_predictions(self, idx, labeled, unlabeled):
        l1 = self.stochastic_augmentation(labeled)
        u1 = self.stochastic_augmentation(unlabeled)

        branch1 = torch.cat([l1, u1])

        self.pred1 = self.model(branch1)

        start_index = idx * self.pred1.size(0)
        end_index = start_index + self.pred1.size(0)
        return self.pred1, self.ensemble_predictions[start_index:end_index]

    def update_ensemble_predictions(self, idx):
        index = idx * self.pred1.size(0)
        self.ema.update_partial(self.pred1, index, self.pred1.shape[0])

    def compute_loss(self, idx, labeled, targets, unlabeled):
        res = super().compute_loss(idx, labeled, targets, unlabeled)

        with torch.no_grad():
            self.update_ensemble_predictions(idx)

        return res
