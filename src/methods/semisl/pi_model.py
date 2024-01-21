import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms
from .semisl_method import SemiSLMethod


AUGMENTATIONS = {
    'crop': torchvision.transforms.RandomCrop(32, padding=4),  # TODO: SIZE MUST DEPEND ON DATASET
    'flip': torchvision.transforms.RandomHorizontalFlip()
}


class PiModel(SemiSLMethod):
    def __init__(self, max_unsupervised_weight, augmentations):
        self.supervised_loss = CrossEntropyLoss()
        self.unsupervised_loss = MSELoss(reduction='sum')
        self.max_unsupervised_weight = max_unsupervised_weight
        self.augmentations = [AUGMENTATIONS[augmentation] for augmentation in augmentations]

    def truncate_batches(self):
        return False

    def on_change_epoch(self, epoch, num_epochs):
        self.unsupervised_weight = min(1.0, epoch / num_epochs + 0.2)  # TODO: CHANGE TO RAMP UP

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

        unsupervised_loss = (unsupervised_loss_1 + unsupervised_loss_2) / (unlabeled_size + labeled_size)  # TODO: DIVIDE BY NUM CLASSES

        total_loss = supervised_loss + self.unsupervised_weight * unsupervised_loss

        loss = {'total': total_loss}
        if labeled is not None:
            loss['supervised'] = supervised_loss
        if unlabeled is not None:
            loss['unsupervised'] = unsupervised_loss

        return labeled_outputs, loss

    def stochastic_augmentation(self, x):
        aug_idx = np.random.randint(len(self.augmentations))
        return self.augmentations[aug_idx](x.clone())

    def augment(self, x):
        x_1 = self.stochastic_augmentation(x)
        x_2 = self.stochastic_augmentation(x)
        return x_1, x_2

    def compute_predictions(self, input):
        aug1, aug2 = self.augment(input)
        predictions1, predictions2 = self.model(aug1), self.model(aug2)
        return self.unsupervised_loss(predictions1, predictions2), predictions1
