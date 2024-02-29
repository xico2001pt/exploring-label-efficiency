import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms.v2 as v2
from .semisl_method import SemiSLMethod
from ...utils.transforms import temperature_sharpening, mixup, GaussianNoise
from ...utils.ramps import linear_rampup


def default_process_targets(targets, num_classes):
    return F.one_hot(targets, num_classes).float()


class MixMatch(SemiSLMethod):
    def __init__(self, alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss, process_targets=None):
        self.beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
        self.max_unsupervised_weight = w_max
        self.unsupervised_weight_fn = linear_rampup(unsupervised_weight_rampup_length)
        self.temperature = temperature
        self.k = k
        self.labeled_transform = labeled_transform
        self.unlabeled_transform = unlabeled_transform
        self.supervised_loss = supervised_loss
        self.unsupervised_loss = unsupervised_loss
        if process_targets is None:
            process_targets = default_process_targets
        self.process_targets = process_targets

    def on_start_train(self, train_data):
        self.num_classes = train_data.num_classes

    def on_start_epoch(self, epoch):
        epoch = epoch - 1
        self.unsupervised_weight = self.unsupervised_weight_fn(epoch) * self.max_unsupervised_weight

    def compute_loss(self, idx, labeled, targets, unlabeled):
        labeled, targets, unlabeled, preds = self.pseudo_labelling(labeled, targets, unlabeled)

        inputs = torch.cat([labeled, unlabeled], dim=0)
        outputs = self.model(inputs)
        del inputs

        labeled_outputs = outputs[:labeled.size(0)]
        supervised_loss = self.supervised_loss(labeled_outputs, targets)

        unlabeled_outputs = outputs[labeled.size(0):].softmax(dim=-1)
        unsupervised_loss = self.unsupervised_loss(unlabeled_outputs, preds)
        unsupervised_weighted_loss = unsupervised_loss * self.unsupervised_weight

        total_loss = supervised_loss + unsupervised_weighted_loss

        loss = {
            'total': total_loss,
            'supervised': supervised_loss,
            'unsupervised': unsupervised_loss,
            'unsupervised_weighted': unsupervised_weighted_loss
        }
        return labeled_outputs, targets.argmax(dim=1), loss

    def pseudo_labelling(self, labeled, targets, unlabeled):
        targets = self.process_targets(targets, self.num_classes)
        labeled, targets = self.labeled_transform(labeled, targets)

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
        indices = torch.randperm(all_inputs.size(0)).to(all_inputs.device)
        all_inputs = all_inputs[indices]
        all_targets = all_targets[indices]

        sep_idx = labeled.size(0)
        labeled, targets = self.mixup(labeled, all_inputs[:sep_idx], targets, all_targets[:sep_idx])
        unlabeled, preds = self.mixup(unlabeled, all_inputs[sep_idx:], preds, all_targets[sep_idx:])

        return labeled, targets, unlabeled, preds

    def mixup(self, x1, x2, y1, y2):
        lam = self.beta_distribution.sample((x1.size(0),)).to(x1.device)
        lam[lam < 0.5] = 1 - lam[lam < 0.5]  # MixMatch needs more weight on the first sample
        return mixup(x1, x2, y1, y2, lam)


def MixMatchCIFAR10(alpha, w_max, unsupervised_weight_rampup_length, temperature, k):
    labeled_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
    ])
    unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4),
        v2.RandomHorizontalFlip(),
        GaussianNoise(),
    ])
    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = MSELoss()
    return MixMatch(alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss)


def MixMatchSVHN(alpha, w_max, unsupervised_weight_rampup_length, temperature, k):
    labeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4),
    ])
    unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4),
        GaussianNoise(),
    ])
    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = MSELoss()
    return MixMatch(alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss)


def MixMatchCityscapesSeg(alpha, w_max, unsupervised_weight_rampup_length, temperature, k):
    labeled_transform = v2.Compose([
        v2.RandomAutocontrast(p=0.5),
    ])
    unlabeled_transform = v2.Compose([
        v2.RandomAutocontrast(p=0.5),
        GaussianNoise(),
    ])
    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = MSELoss()

    def process_targets(targets, num_classes):
        # TODO: Convert to tv tensor
        targets = F.one_hot(targets, num_classes).float()
        return targets.permute(0, 3, 1, 2)
    return MixMatch(alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss, process_targets)
