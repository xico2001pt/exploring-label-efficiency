import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms.v2 as v2
from ..semisl_method import SemiSLMethod
from ....utils.ramps import exp_rampup


class PiModel(SemiSLMethod):
    def __init__(self, w_max, unsupervised_weight_rampup_length):
        self.supervised_loss = CrossEntropyLoss(reduction='mean')
        self.unsupervised_loss = MSELoss(reduction='mean')
        self.max_unsupervised_weight = w_max
        self.unsupervised_weight_fn = exp_rampup(unsupervised_weight_rampup_length)

    def on_start_train(self, train_data):
        self.max_unsupervised_weight *= train_data.dataset_size["labeled"] / train_data.dataset_size["total"]
        self.num_classes = train_data.num_classes

        input_size = train_data.input_size[-2:]
        self.augmentations = v2.Compose([
            v2.RandomCrop(input_size, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ])

    def on_start_epoch(self, epoch):
        epoch = epoch - 1
        self.unsupervised_weight = self.unsupervised_weight_fn(epoch) * self.max_unsupervised_weight if epoch > 0 else 0.0

    def get_predictions(self, idx, labeled, unlabeled):
        l1, l2 = self.augment(labeled)
        u1, u2 = self.augment(unlabeled)

        branch1, branch2 = torch.cat([l1, u1]), torch.cat([l2, u2])

        pred1 = self.model(branch1)
        with torch.no_grad():
            pred2 = self.model(branch2)

        return pred1, pred2

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

    def stochastic_augmentation(self, x):
        return self.augmentations(x)

    def augment(self, x):
        x_1 = self.stochastic_augmentation(x)
        x_2 = self.stochastic_augmentation(x)
        return x_1, x_2
