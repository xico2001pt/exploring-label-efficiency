import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from .semisl_method import SemiSLMethod
from ...core.losses import CrossEntropyWithLogitsLoss
from ...utils.transforms import temperature_sharpening, mixup, GaussianNoise, InvariantRandAugment
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

        unlabeled_outputs = outputs[labeled.size(0):].softmax(dim=1)
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

        unlabeled = [self.unlabeled_transform(unlabeled) for _ in range(self.k)]

        with torch.no_grad():
            preds = [self.model(unlabeled[k]).softmax(dim=1) for k in range(self.k)]
            preds = temperature_sharpening(sum(preds) / self.k, self.temperature)
            preds = preds.detach()

        all_inputs = torch.cat([labeled] + unlabeled, dim=0)
        all_targets = torch.cat([targets] + [preds] * self.k, dim=0)

        mixed_inputs, mixed_targets = self.mixup(all_inputs, all_targets)

        sep_idx = labeled.size(0)
        labeled = mixed_inputs[:sep_idx]
        targets = mixed_targets[:sep_idx]
        unlabeled = mixed_inputs[sep_idx:]
        preds = mixed_targets[sep_idx:]

        return labeled, targets, unlabeled, preds

    def mixup(self, all_inputs, all_targets):
        lam = self.beta_distribution.sample().item()
        lam = max(lam, 1 - lam)

        indices = torch.randperm(all_inputs.size(0)).to(all_inputs.device)
        input_a, input_b = all_inputs, all_inputs[indices]
        target_a, target_b = all_targets, all_targets[indices]

        return mixup(input_a, input_b, target_a, target_b, lam)


def MixMatchCIFAR10(alpha, w_max, unsupervised_weight_rampup_length, temperature, k):
    labeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
    ])
    unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
        GaussianNoise(),
    ])
    supervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    unsupervised_loss = MSELoss()
    return MixMatch(alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss)


def MixMatchSVHN(alpha, w_max, unsupervised_weight_rampup_length, temperature, k):
    labeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    ])
    unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        GaussianNoise(),
    ])
    supervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    unsupervised_loss = MSELoss()
    return MixMatch(alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss)


def MixMatchCityscapesSeg(alpha, w_max, unsupervised_weight_rampup_length, temperature, k):
    labeled_transform = v2.Compose([
        v2.RandomCrop((512, 512), padding=64, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
    ])
    unlabeled_transform = InvariantRandAugment(2, 10)
    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = MSELoss()

    def process_targets(targets, num_classes):
        targets = F.one_hot(targets, num_classes).float()
        targets = targets.permute(0, 3, 1, 2)
        return tv_tensors.Mask(targets)
    return MixMatch(alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss, process_targets)
