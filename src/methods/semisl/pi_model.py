import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms.v2 as v2
from .semisl_method import SemiSLMethod
from ...utils.ramps import exp_rampup
from ...utils.transforms import InvariantRandAugment


class PiModel(SemiSLMethod):
    def __init__(self, w_max, unsupervised_weight_rampup_length, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss):
        self.supervised_loss = supervised_loss
        self.unsupervised_loss = unsupervised_loss
        self.max_unsupervised_weight = w_max
        self.unsupervised_weight_fn = exp_rampup(unsupervised_weight_rampup_length)
        self.labeled_transform = labeled_transform
        self.unlabeled_transform = unlabeled_transform

    def on_start_train(self, train_data):
        self.max_unsupervised_weight *= train_data.dataset_size["labeled"] / train_data.dataset_size["total"]
        self.num_classes = train_data.num_classes

    def on_start_epoch(self, epoch):
        epoch = epoch - 1
        self.unsupervised_weight = self.unsupervised_weight_fn(epoch) * self.max_unsupervised_weight if epoch > 0 else 0.0

    def compute_loss(self, idx, labeled, targets, unlabeled):
        pred1, pred2 = self.get_predictions(idx, labeled, unlabeled)

        labeled_outputs = pred1[:labeled.size(0)]
        supervised_loss = self.supervised_loss(labeled_outputs, targets)

        unsupervised_loss = self.unsupervised_loss(pred1, pred2)
        unsupervised_weighted_loss = self.unsupervised_weight * unsupervised_loss

        total_loss = supervised_loss + unsupervised_weighted_loss

        loss = {
            'total': total_loss,
            'supervised': supervised_loss,
            'unsupervised': unsupervised_loss,
            'unsupervised_weighted': unsupervised_weighted_loss
        }
        return labeled_outputs, targets, loss

    def get_predictions(self, idx, labeled, unlabeled):
        l1, l2 = self.augment(labeled, True)
        u1, u2 = self.augment(unlabeled, False)

        branch1, branch2 = torch.cat([l1, u1]), torch.cat([l2, u2])

        pred1 = self.model(branch1)
        with torch.no_grad():
            pred2 = self.model(branch2)

        return pred1.softmax(dim=1), pred2.softmax(dim=1)

    def labeled_augmentation(self, x):
        return self.labeled_transform(x)

    def unlabeled_augmentation(self, x):
        return self.unlabeled_transform(x)

    def augment(self, x, labeled=True):
        aug_fn = self.labeled_augmentation if labeled else self.unlabeled_augmentation
        x_1 = aug_fn(x)
        x_2 = aug_fn(x)
        return x_1, x_2


def PiModelCIFAR10(w_max, unsupervised_weight_rampup_length):
    transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandomHorizontalFlip(p=0.5),
    ])
    supervised_loss = CrossEntropyLoss(reduction='mean')
    unsupervised_loss = MSELoss(reduction='mean')
    return PiModel(w_max, unsupervised_weight_rampup_length, transform, transform, supervised_loss, unsupervised_loss)


def PiModelSVHN(w_max, unsupervised_weight_rampup_length):
    transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    ])
    supervised_loss = CrossEntropyLoss(reduction='mean')
    unsupervised_loss = MSELoss(reduction='mean')
    return PiModel(w_max, unsupervised_weight_rampup_length, transform, transform, supervised_loss, unsupervised_loss)


def PiModelCityscapesSeg(w_max, unsupervised_weight_rampup_length):
    labeled_transform = v2.Identity()
    unlabeled_transform = v2.Compose([
        InvariantRandAugment(2, 10),
    ])
    supervised_loss = CrossEntropyLoss(reduction='mean')
    unsupervised_loss = CrossEntropyLoss(reduction='mean')

    return PiModel(w_max, unsupervised_weight_rampup_length, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss)
