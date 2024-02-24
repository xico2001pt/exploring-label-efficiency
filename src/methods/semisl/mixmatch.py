import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms.v2 as v2
from .semisl_method import SemiSLMethod
from ...utils.functional import temperature_sharpening, mixup


class MixMatch(SemiSLMethod):
    def __init__(self, alpha, lam_u, temperature, k, transform, supervised_loss, unsupervised_loss):
        self.beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
        self.lam_u = lam_u
        self.temperature = temperature
        self.k = k
        self.augmentations = transform
        self.supervised_loss = supervised_loss
        self.unsupervised_loss = unsupervised_loss

    def on_start_train(self, train_data):
        self.num_classes = train_data.num_classes

    def compute_loss(self, idx, labeled, targets, unlabeled):
        labeled, targets, unlabeled, preds = self.pseudo_labelling(labeled, targets, unlabeled)

        labeled_outputs = self.model(labeled)
        supervised_loss = self.supervised_loss(labeled_outputs, targets)

        unsupervised_loss = self.unsupervised_loss(preds, self.model(unlabeled))
        unsupervised_weighted_loss = unsupervised_loss * self.lam_u

        total_loss = supervised_loss + unsupervised_weighted_loss

        loss = {
            'total': total_loss,
            'supervised': supervised_loss,
            'unsupervised': unsupervised_loss,
            'unsupervised_weighted': unsupervised_weighted_loss
        }
        return labeled_outputs, targets, loss

    def pseudo_labelling(self, labeled, targets, unlabeled):
        # TODO: Cast targets tv tensor
        labeled, targets = self.stochastic_augmentation((labeled, targets))
        targets = F.one_hot(targets, self.num_classes)
        targets = targets.float()

        with torch.no_grad():
            unlabeled = [self.stochastic_augmentation(unlabeled) for _ in range(self.k)]

            preds = [self.model(unlabeled[k]).softmax(dim=-1) for k in range(self.k)]
            preds = temperature_sharpening(sum(preds) / self.k, self.temperature)
            preds = preds.detach()

            unlabeled = torch.cat(unlabeled)

        all_inputs = torch.cat([labeled, unlabeled])
        all_targets = torch.cat([targets, preds])

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

    def stochastic_augmentation(self, x):
        return self.augmentations(x)


def MixMatchCIFAR10(alpha, lam_u, temperature, k):
    transform = v2.Compose([
        v2.Identity(),
    ])
    supervised_loss = CrossEntropyLoss(reduction='mean')
    unsupervised_loss = MSELoss(reduction='mean')
    return MixMatch(alpha, lam_u, temperature, k, transform, supervised_loss, unsupervised_loss)
