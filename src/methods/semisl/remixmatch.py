import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torchvision.transforms.v2 as v2
from .semisl_method import SemiSLMethod
from ...core.losses import CrossEntropyWithLogitsLoss
from ...utils.transforms import temperature_sharpening, mixup, GaussianNoise
from ...utils.ramps import linear_rampup


def default_process_targets(targets, num_classes):
    return F.one_hot(targets, num_classes).float()


def normalize(x):
    return temperature_sharpening(x, 1)


class ReMixMatch(SemiSLMethod):
    def __init__(self, alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, rotation_loss, process_targets=default_process_targets):
        self.alpha = alpha
        self.w_max = w_max
        self.unsupervised_weight_rampup_length = unsupervised_weight_rampup_length
        self.temperature = temperature
        self.k = k
        self.labeled_transform = labeled_transform
        self.weak_unlabeled_transform = weak_unlabeled_transform
        self.strong_unlabeled_transform = strong_unlabeled_transform
        self.supervised_loss = supervised_loss
        self.unsupervised_loss = unsupervised_loss
        self.rotation_loss = rotation_loss
        if process_targets is None:
            process_targets = default_process_targets
        self.process_targets = process_targets

    def on_start_train(self, train_data):
        self.num_classes = train_data.num_classes

    def on_start_epoch(self, epoch):
        epoch = epoch - 1
        self.unsupervised_weight = linear_rampup(epoch, self.unsupervised_weight_rampup_length) * self.w_max

    def compute_loss(self, idx, labeled, targets, unlabeled):
        labeled, targets, unlabeled, unlabeled1, preds = self.pseudo_labelling(labeled, targets, unlabeled)

        inputs = torch.cat([labeled, unlabeled, unlabeled1], dim=0)
        outputs = self.model(inputs)
        del inputs

        unlabeled_idx = labeled.size(0)
        unlabeled1_idx = unlabeled_idx + unlabeled.size(0)

        labeled_outputs = outputs[:unlabeled_idx]
        supervised_loss = self.supervised_loss(labeled_outputs, targets)

        unlabeled_outputs = outputs[unlabeled_idx:unlabeled1_idx].softmax(dim=-1)
        unsupervised_loss = self.unsupervised_loss(unlabeled_outputs, preds)
        unsupervised_weighted_loss = unsupervised_loss * self.unsupervised_weight

        # TODO: unsupervised1_loss and rotation_loss

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
            strong_unlabeled = [self.strong_unlabeled_transform(unlabeled) for _ in range(self.k)]
            weak_unlabeled = self.weak_unlabeled_transform(unlabeled)

            preds = self.model(weak_unlabeled).softmax(dim=-1)
            # TODO: preds = normalize(preds * p(y) / p~(y))
            preds = normalize(temperature_sharpening(preds, self.temperature))
            preds = preds.detach()

        unlabeled = torch.cat(strong_unlabeled + [weak_unlabeled])
        preds = torch.cat([preds] * (self.k + 1))

        all_inputs = torch.cat([labeled, unlabeled], dim=0)
        all_targets = torch.cat([targets, preds], dim=0)

        unlabeled1 = None  # TODO: get unlabeled1

        mixed_inputs, mixed_targets = self.mixup(all_inputs, all_targets)

        sep_idx = labeled.size(0)
        labeled = mixed_inputs[:sep_idx]
        targets = mixed_targets[:sep_idx]
        unlabeled = mixed_inputs[sep_idx:]
        preds = mixed_targets[sep_idx:]

        return labeled, targets, unlabeled, unlabeled1, preds

    def mixup(self, all_inputs, all_targets):
        lam = self.beta_distribution.sample().item()
        lam = max(lam, 1 - lam)

        indices = torch.randperm(all_inputs.size(0)).to(all_inputs.device)
        input_a, input_b = all_inputs, all_inputs[indices]
        target_a, target_b = all_targets, all_targets[indices]

        return mixup(input_a, input_b, target_a, target_b, lam)


def ReMixMatchCIFAR10(alpha, w_max, unsupervised_weight_rampup_length, temperature, k):
    labeled_transform = v2.Identity()  # TODO
    weak_unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
    ])
    strong_unlabeled_transform = v2.Identity()  # TODO
    supervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    unsupervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    rotation_loss = CrossEntropyLoss()
    return ReMixMatch(alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, rotation_loss)
