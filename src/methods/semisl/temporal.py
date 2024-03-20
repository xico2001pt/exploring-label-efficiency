import torch
import torchvision.transforms.v2 as v2
from torch.nn import CrossEntropyLoss, MSELoss
from .pi_model import PiModel
from ...utils.structures import EMA


class TemporalEnsembling(PiModel):
    def __init__(self, w_max, unsupervised_weight_rampup_length, accumlation_decay, transform, supervised_loss, unsupervised_loss):
        super().__init__(w_max, unsupervised_weight_rampup_length, transform, supervised_loss, unsupervised_loss)
        self.accumlation_decay = accumlation_decay

    def on_start_train(self, train_data):
        super().on_start_train(train_data)

        labeled_size = train_data.dataset_size["labeled"]
        unlabeled_size = train_data.dataset_size["unlabeled"]
        self.labeled_ema = EMA(self.accumlation_decay, torch.zeros(size=(labeled_size, self.num_classes), requires_grad=False, device=train_data.device, dtype=torch.float32))
        self.unlabeled_ema = EMA(self.accumlation_decay, torch.zeros(size=(unlabeled_size, self.num_classes), requires_grad=False, device=train_data.device, dtype=torch.float32))
        self.labeled_ensemble_predictions = self.labeled_ema.get_value().detach().clone()
        self.unlabeled_ensemble_predictions = self.unlabeled_ema.get_value().detach().clone()

    def on_end_epoch(self, epoch):
        super().on_end_epoch(epoch)

        with torch.no_grad():
            self.labeled_ensemble_predictions = self.labeled_ema.get_value().detach().clone() / (1 - self.accumlation_decay ** epoch)
            self.unlabeled_ensemble_predictions = self.unlabeled_ema.get_value().detach().clone() / (1 - self.accumlation_decay ** epoch)

    def get_predictions(self, idx, labeled, unlabeled):
        l1 = self.stochastic_augmentation(labeled)
        u1 = self.stochastic_augmentation(unlabeled)

        branch1 = torch.cat([l1, u1])

        self.pred1 = self.model(branch1)

        labeled_start_index = idx[0] * l1.size(0)
        labeled_end_index = labeled_start_index + l1.size(0)
        unlabeled_start_index = idx[1] * u1.size(0)
        unlabeled_end_index = unlabeled_start_index + u1.size(0)

        ensemble_predictions = torch.cat([self.labeled_ensemble_predictions[labeled_start_index:labeled_end_index], self.unlabeled_ensemble_predictions[unlabeled_start_index:unlabeled_end_index]])
        return self.pred1, ensemble_predictions

    def update_ensemble_predictions(self, idx, labeled_size, unlabeled_size):
        labeled_index = idx[0] * labeled_size
        self.labeled_ema.update_partial(self.pred1[:labeled_size], labeled_index, labeled_size)

        unlabeled_index = idx[1] * unlabeled_size
        self.unlabeled_ema.update_partial(self.pred1[labeled_size:], unlabeled_index, unlabeled_size)

    def compute_loss(self, idx, labeled, targets, unlabeled):
        res = super().compute_loss(idx, labeled, targets, unlabeled)

        with torch.no_grad():
            self.update_ensemble_predictions(idx, labeled.size(0), unlabeled.size(0))

        return res


def TemporalEnsemblingCIFAR10(w_max, unsupervised_weight_rampup_length, accumlation_decay):
    transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandomHorizontalFlip(p=0.5),
    ])
    supervised_loss = CrossEntropyLoss(reduction='mean')
    unsupervised_loss = MSELoss(reduction='mean')
    return TemporalEnsembling(w_max, unsupervised_weight_rampup_length, accumlation_decay, transform, supervised_loss, unsupervised_loss)


def TemporalEnsemblingSVHN(w_max, unsupervised_weight_rampup_length, accumlation_decay):
    transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    ])
    supervised_loss = CrossEntropyLoss(reduction='mean')
    unsupervised_loss = MSELoss(reduction='mean')
    return TemporalEnsembling(w_max, unsupervised_weight_rampup_length, accumlation_decay, transform, supervised_loss, unsupervised_loss)
