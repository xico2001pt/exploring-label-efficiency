import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.tv_tensors as tv_tensors
import torchvision.transforms.v2 as v2
from ..semisl_method import SemiSLMethod
from ....utils.ramps import exp_rampup


class PiModelSeg(SemiSLMethod):
    def __init__(self, w_max, unsupervised_weight_rampup_length):
        self.supervised_loss = CrossEntropyLoss(reduction='mean')
        self.unsupervised_loss = MSELoss(reduction='mean')
        self.max_unsupervised_weight = w_max
        self.unsupervised_weight_fn = exp_rampup(unsupervised_weight_rampup_length)

    def on_start_train(self, train_data):
        self.max_unsupervised_weight *= train_data.dataset_size["labeled"] / train_data.dataset_size["total"]
        self.num_classes = train_data.num_classes

        input_size = train_data.input_size[-2:]
        self.common_augmentations = v2.Compose([
            v2.RandomCrop(input_size, padding=4),
            v2.RandomHorizontalFlip(p=0.5)
        ])
        self.individual_augmentations = v2.Compose([
            #v2.Identity(),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            #v2.RandomAutocontrast(p=0.5),
            #v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            #v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def on_start_epoch(self, epoch):
        epoch = epoch - 1
        self.unsupervised_weight = self.unsupervised_weight_fn(epoch) * self.max_unsupervised_weight if epoch > 0 else 0.0

    def get_predictions(self, idx, labeled, targets, unlabeled):
        labeled, targets = tv_tensors.Image(labeled), tv_tensors.Mask(targets)
        unlabeled = tv_tensors.Image(unlabeled)

        l1, l2, targets = self.augment_labeled(labeled, targets)
        u1, u2 = self.augment_unlabeled(unlabeled)

        branch1, branch2 = torch.cat([l1, u1]), torch.cat([l2, u2])

        pred1 = self.model(branch1)
        with torch.no_grad():
            pred2 = self.model(branch2)

        return pred1, pred2, targets

    def compute_loss(self, idx, labeled, targets, unlabeled):
        pred1, pred2, targets = self.get_predictions(idx, labeled, targets, unlabeled)

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

    def augment_labeled(self, labeled, targets):
        labeled, targets = self.common_augmentations(labeled, targets)
        labeled_1 = self.individual_augmentations(labeled)
        labeled_2 = self.individual_augmentations(labeled)
        return labeled_1, labeled_2, targets

    def augment_unlabeled(self, unlabeled):
        unlabeled = self.common_augmentations(unlabeled)
        unlabeled_1 = self.individual_augmentations(unlabeled)
        unlabeled_2 = self.individual_augmentations(unlabeled)
        return unlabeled_1, unlabeled_2
