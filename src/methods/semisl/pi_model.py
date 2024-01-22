import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms
from .semisl_method import SemiSLMethod
from ...utils.ramps import exp_rampup


AUGMENTATIONS = {
    'crop': torchvision.transforms.RandomCrop(32, padding=4),  # TODO: SIZE MUST DEPEND ON DATASET
    'flip': torchvision.transforms.RandomHorizontalFlip(p=0.5),
}


class PiModel(SemiSLMethod):
    def __init__(self, max_unsupervised_weight, unsupervised_weight_rampup_length, augmentations):
        self.supervised_loss = CrossEntropyLoss(reduction='mean')
        self.unsupervised_loss = MSELoss(reduction='sum')
        self.max_unsupervised_weight = max_unsupervised_weight
        self.unsupervised_weight_fn = exp_rampup(unsupervised_weight_rampup_length)
        self.augmentations = [AUGMENTATIONS[augmentation] for augmentation in augmentations]

    def truncate_batches(self):
        return False

    def on_change_epoch(self, epoch):
        self.unsupervised_weight = self.unsupervised_weight_fn(epoch) * self.max_unsupervised_weight if epoch > 0 else 0.0

    def compute_loss(self, labeled, targets, unlabeled):
        if unlabeled is not None:
            p1, p2 = self.compute_predictions(unlabeled)
            unsupervised_loss_1 = self.unsupervised_loss(p1, p2)
            unlabeled_size = unlabeled.size(0)
        else:
            unsupervised_loss_1 = 0.0
            unlabeled_size = 0

        if labeled is not None:
            p1, p2 = self.compute_predictions(labeled)
            labeled_outputs = p1
            supervised_loss = self.supervised_loss(labeled_outputs, targets)
            unsupervised_loss_2 = self.unsupervised_loss(p1, p2)
            labeled_size = labeled.size(0)
        else:
            labeled_outputs = None
            supervised_loss = 0.0
            unsupervised_loss_2 = 0.0
            labeled_size = 0

        total_size = labeled_size + unlabeled_size  # Never zero

        unsupervised_loss = (unsupervised_loss_1 + unsupervised_loss_2) / float(total_size / 10)  # TODO: NUM CLASSES VARIABLE
        unsupervised_weighted_loss = self.unsupervised_weight * unsupervised_loss

        total_loss = supervised_loss + unsupervised_weighted_loss

        loss = {'total': total_loss}
        if labeled is not None:
            loss['supervised'] = supervised_loss
        if unlabeled is not None:
            loss['unsupervised'] = unsupervised_loss
            loss['unsupervised_weighted'] = unsupervised_weighted_loss

        return labeled_outputs, loss

    def stochastic_augmentation(self, x):
        for augmentation in self.augmentations:
            x = augmentation(x)
        return x

    def augment(self, x):
        x_1 = self.stochastic_augmentation(x.clone())
        x_2 = self.stochastic_augmentation(x.clone())
        return x_1, x_2

    def compute_predictions(self, input):
        aug1, aug2 = self.augment(input)
        predictions1 = self.model(aug1)
        with torch.no_grad():
            predictions2 = self.model(aug2)
        return predictions1, predictions2
