import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms.v2 as v2
from .semisl_method import SemiSLMethod
from ...utils.functional import temperature_sharpening, mixup
from ...utils.ramps import linear_rampup


class MixMatch(SemiSLMethod):
    def __init__(self, alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss):
        self.beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
        self.max_unsupervised_weight = w_max
        self.unsupervised_weight_fn = linear_rampup(unsupervised_weight_rampup_length)
        self.temperature = temperature
        self.k = k
        self.labeled_transform = labeled_transform
        self.unlabeled_transform = unlabeled_transform
        self.supervised_loss = supervised_loss
        self.unsupervised_loss = unsupervised_loss

    def on_start_train(self, train_data):
        self.num_classes = train_data.num_classes

    def on_start_epoch(self, epoch):
        epoch = epoch - 1
        self.unsupervised_weight = self.unsupervised_weight_fn(epoch) * self.max_unsupervised_weight

    def compute_loss(self, idx, labeled, targets, unlabeled):
        labeled, targets, unlabeled, preds = self.pseudo_labelling(labeled, targets, unlabeled)

        labeled_outputs = self.model(labeled)
        supervised_loss = self.supervised_loss(labeled_outputs, targets)

        unsupervised_loss = self.unsupervised_loss(preds, self.model(unlabeled))
        unsupervised_weighted_loss = unsupervised_loss * self.unsupervised_weight

        total_loss = supervised_loss + unsupervised_weighted_loss

        loss = {
            'total': total_loss,
            'supervised': supervised_loss,
            'unsupervised': unsupervised_loss,
            'unsupervised_weighted': unsupervised_weighted_loss
        }
        return labeled_outputs, targets.argmax(dim=-1), loss

    def pseudo_labelling(self, labeled, targets, unlabeled):
        # TODO: Cast targets tv tensor
        labeled, targets = self.labeled_transform(labeled, targets)
        targets = F.one_hot(targets, self.num_classes)
        targets = targets.float()

        with torch.no_grad():
            unlabeled = [self.unlabeled_transform(unlabeled) for _ in range(self.k)]

            preds = [self.model(unlabeled[k]).softmax(dim=-1) for k in range(self.k)]
            preds = temperature_sharpening(sum(preds) / self.k, self.temperature)
            preds = preds.detach()

            unlabeled = torch.cat(unlabeled)
            preds = torch.cat([preds] * self.k)

        all_inputs = torch.cat([labeled, unlabeled], dim=0)
        all_targets = torch.cat([targets, preds], dim=0)

        # Shuffle (make sure to shuffle the inputs and targets in the same way)
        indices = torch.randperm(all_inputs.size(0))
        all_inputs = all_inputs[indices]
        all_targets = all_targets[indices]

        sep_idx = labeled.size(0)
        labeled, targets = self.mixup(labeled, all_inputs[:sep_idx], targets, all_targets[:sep_idx])
        unlabeled, preds = self.mixup(unlabeled, all_inputs[sep_idx:], preds, all_targets[sep_idx:])

        return labeled, targets, unlabeled, preds

    def mixup(self, x1, x2, y1, y2):
        lam = self.beta_distribution.sample().item()

        lam = max(lam, 1 - lam)  # MixMatch needs more weight on the first sample

        return mixup(x1, x2, y1, y2, lam)


def MixMatchCIFAR10(alpha, w_max, unsupervised_weight_rampup_length, temperature, k):
    transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
    ])
    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = MSELoss()
    return MixMatch(alpha, w_max, unsupervised_weight_rampup_length, temperature, k, transform, transform, supervised_loss, unsupervised_loss)