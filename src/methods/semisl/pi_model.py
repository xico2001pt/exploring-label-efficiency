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
        self.unsupervised_loss = MSELoss(reduction='mean')
        self.max_unsupervised_weight = max_unsupervised_weight
        self.unsupervised_weight_fn = exp_rampup(unsupervised_weight_rampup_length)
        self.augmentations = [AUGMENTATIONS[augmentation] for augmentation in augmentations]

    def truncate_batches(self):
        return False

    def on_change_epoch(self, epoch):
        epoch = epoch - 1
        self.unsupervised_weight = self.unsupervised_weight_fn(epoch) * self.max_unsupervised_weight if epoch > 0 else 0.0

    def compute_loss(self, labeled, targets, unlabeled):
        if labeled is not None:
            l1, l2 = self.augment(labeled)
        else:
            l1 = l2 = torch.empty(0).to(unlabeled.device)
            supervised_loss = 0.0
            labeled_outputs = None

        if unlabeled is not None:
            u1, u2 = self.augment(unlabeled)
        else:
            u1 = u2 = torch.empty(0).to(labeled.device)

        cat1, cat2 = torch.cat([l1, u1]), torch.cat([l2, u2])

        out1 = self.model(cat1)

        with torch.no_grad():
            out2 = self.model(cat2)

        if labeled is not None:
            labeled_outputs = out1[:labeled.size(0)]
            supervised_loss = self.supervised_loss(labeled_outputs, targets)

        unsupervised_loss = self.unsupervised_loss(out1, out2)
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
