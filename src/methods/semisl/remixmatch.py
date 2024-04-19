import torch
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from .semisl_method import SemiSLMethod
from ...core.losses import CrossEntropyWithLogitsLoss
from ...utils.structures import TensorMovingAverage
from ...utils.transforms import temperature_sharpening, mixup, InvariantRandAugment
from ...utils.ramps import linear_rampup
from ...utils.utils import classes_mean, backbone_getter


def default_process_targets(targets, num_classes):
    return F.one_hot(targets, num_classes).float()


def normalize(x):
    return temperature_sharpening(x, 1)


class ReMixMatch(SemiSLMethod):
    def __init__(self, alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k, gt_labels, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, rotation_loss, process_targets=default_process_targets):
        self.beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
        self.wu_max = wu_max
        self.wu1_max = wu1_max
        self.wr = wr
        self.unsupervised_weight_fn = linear_rampup(unsupervised_weight_rampup_length)
        self.temperature = temperature
        self.k = k
        self.gt_labels = gt_labels
        self.labeled_transform = labeled_transform
        self.weak_unlabeled_transform = weak_unlabeled_transform
        self.strong_unlabeled_transform = strong_unlabeled_transform
        self.supervised_loss = supervised_loss
        self.unsupervised_loss = unsupervised_loss
        self.rotation_loss = rotation_loss
        self.rotation_classifier = None
        if process_targets is None:
            process_targets = default_process_targets
        self.process_targets = process_targets

    def set_model(self, model):
        super().set_model(model)

        if self.wr <= 0:
            return

        def on_forward(m, inputs, output):
            if isinstance(output, OrderedDict):
                output = output['out']

            if self.rotation_classifier is None:
                self.init_rotation_classifier(output.size(1))

            if len(output.shape) > 2:  # TODO: Why is this necessary?
                output = output.mean([2, 3])

            self.embeddings = output

        backbone = backbone_getter(model)
        self.rot_hook = backbone.register_forward_hook(on_forward)

    def on_start_train(self, train_data):
        self.num_classes = train_data.num_classes
        self.preds_moving_average = TensorMovingAverage(128, self.num_classes, train_data.device)
        if self.gt_labels is not None:
            self.gt_labels = self.gt_labels.to(train_data.device)
        if self.wr > 0:
            self.device = train_data.device
            self.optimizer = train_data.optimizer

    def on_start_epoch(self, epoch):
        epoch = epoch - 1
        self.unsupervised_weight = self.unsupervised_weight_fn(epoch) * self.wu_max
        self.unsupervised1_weight = self.unsupervised_weight_fn(epoch) * self.wu1_max

    def on_end_train(self, train_data):
        if self.wr <= 0:
            return

        # Delete param group
        self.optimizer.param_groups = self.optimizer.param_groups[:-1]

        # Delete hook
        self.rot_hook.remove()

    def init_rotation_classifier(self, num_features):
        self.rotation_classifier = torch.nn.Linear(num_features, 4).to(self.device)
        self.optimizer.add_param_group({'params': self.rotation_classifier.parameters()})
        self.rotation_classifier.train()

    def create_rotation_task(self, inputs):
        split = inputs.size(0) // 4
        x1, x2, x3, x4 = inputs[:split], inputs[split:2*split], inputs[2*split:3*split], inputs[3*split:]
        x1 = x1.rot90(1, [2, 3])
        x2 = x2.rot90(2, [2, 3])
        x3 = x3.rot90(3, [2, 3])

        targets = torch.zeros(split, dtype=torch.long)
        targets = torch.cat([targets, targets + 1, targets + 2, targets + 3])
        targets = torch.cat([targets, torch.zeros(len(inputs) - len(targets), dtype=torch.long) + 3])
        targets = targets.to(inputs.device)

        inputs = torch.cat([x1, x2, x3, x4], dim=0)
        return inputs, targets

    def compute_loss(self, idx, labeled, targets, unlabeled):
        labeled, targets, unlabeled, unlabeled1, preds = self.pseudo_labelling(labeled, targets, unlabeled)

        inputs = torch.cat([labeled, unlabeled, unlabeled1], dim=0)
        outputs = self.model(inputs)
        del inputs

        unlabeled_idx = labeled.size(0)
        unlabeled1_idx = unlabeled_idx + unlabeled.size(0)

        labeled_outputs = outputs[:unlabeled_idx]
        supervised_loss = self.supervised_loss(labeled_outputs, targets)

        unlabeled_outputs = outputs[unlabeled_idx:unlabeled1_idx]#.softmax(dim=1)
        unsupervised_loss = self.unsupervised_loss(unlabeled_outputs, preds)
        unsupervised_weighted_loss = unsupervised_loss * self.unsupervised_weight

        unlabeled1_outputs = outputs[unlabeled1_idx:]#.softmax(dim=1)
        unsupervised1_loss = self.unsupervised_loss(unlabeled1_outputs, preds[:unlabeled1_outputs.size(0)])
        unsupervised1_weighted_loss = unsupervised1_loss * self.unsupervised1_weight

        total_loss = supervised_loss + unsupervised_weighted_loss + unsupervised1_weighted_loss

        loss = {
            'total': total_loss,
            'supervised': supervised_loss,
            'unsupervised': unsupervised_loss,
            'unsupervised_weighted': unsupervised_weighted_loss,
            'unsupervised1': unsupervised1_loss,
            'unsupervised1_weighted': unsupervised1_weighted_loss,
        }

        if self.wr > 0:
            rot_inputs, rot_targets = self.create_rotation_task(unlabeled1)
            self.model(rot_inputs)  # Collect the embeddings
            rot_outputs = self.rotation_classifier(self.embeddings)
            rotation_loss = self.rotation_loss(rot_outputs, rot_targets)
            rotation_weighted_loss = rotation_loss * self.wr

            total_loss += rotation_weighted_loss

            loss['total'] = total_loss
            loss['rotation'] = rotation_loss
            loss['rotation_weighted'] = rotation_weighted_loss

        return labeled_outputs, targets.argmax(dim=1), loss

    def pseudo_labelling(self, labeled, targets, unlabeled):
        targets = self.process_targets(targets, self.num_classes)
        labeled, targets = self.labeled_transform(labeled, targets)

        with torch.no_grad():
            strong_unlabeled = [self.strong_unlabeled_transform(unlabeled) for _ in range(self.k)]
            weak_unlabeled = self.weak_unlabeled_transform(unlabeled)

            preds = self.model(weak_unlabeled).softmax(dim=1)

            # Distribution alignment #
            gt_labels = self.gt_labels if self.gt_labels is not None else classes_mean(targets)
            preds = normalize(self.apply_distribution_alignment(gt_labels, preds))
            ##########################

            preds = normalize(temperature_sharpening(preds, self.temperature))
            preds = preds.detach()

        unlabeled = torch.cat(strong_unlabeled + [weak_unlabeled])
        preds = torch.cat([preds] * (self.k + 1))

        all_inputs = torch.cat([labeled, unlabeled], dim=0)
        all_targets = torch.cat([targets, preds], dim=0)

        unlabeled1 = strong_unlabeled[0]

        mixed_inputs, mixed_targets = self.mixup(all_inputs, all_targets)

        sep_idx = labeled.size(0)
        labeled = mixed_inputs[:sep_idx]
        targets = mixed_targets[:sep_idx]
        unlabeled = mixed_inputs[sep_idx:]
        preds = mixed_targets[sep_idx:]

        return labeled, targets, unlabeled, unlabeled1, preds

    def apply_distribution_alignment(self, gt_labels, preds):
        ratio = (1e-6 + gt_labels) / (1e-6 + self.preds_moving_average.get_value())
        #preds = preds * ratio.view(-1, 1, 1)  # TODO: Only used in segmentation
        self.preds_moving_average.update(preds)
        return preds

    def mixup(self, all_inputs, all_targets):
        lam = self.beta_distribution.sample().item()
        lam = max(lam, 1 - lam)

        indices = torch.randperm(all_inputs.size(0)).to(all_inputs.device)
        input_a, input_b = all_inputs, all_inputs[indices]
        target_a, target_b = all_targets, all_targets[indices]

        return mixup(input_a, input_b, target_a, target_b, lam)


def ReMixMatchCIFAR10(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k):
    gt_labels = torch.ones(10) / 10
    labeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
    ])
    weak_unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
    ])
    strong_unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
        v2.RandAugment(2, 10),
    ])
    supervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    unsupervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    rotation_loss = CrossEntropyLoss()
    return ReMixMatch(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k, gt_labels, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, rotation_loss)


def ReMixMatchSVHN(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k):
    gt_labels = torch.ones(10) / 10
    labeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    ])
    weak_unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    ])
    strong_unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandAugment(2, 10),
    ])
    supervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    unsupervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    rotation_loss = CrossEntropyLoss()
    return ReMixMatch(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k, gt_labels, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, rotation_loss)


def ReMixMatchCityscapesSeg(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k):
    gt_labels = None
    labeled_transform = v2.Compose([
        v2.RandomCrop((512, 512), padding=64, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
    ])
    weak_unlabeled_transform = v2.Identity()

    strong_unlabeled_transform = v2.Compose([
        InvariantRandAugment(2, 10),
    ])
    supervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    unsupervised_loss = CrossEntropyWithLogitsLoss(return_dict=False)
    rotation_loss = CrossEntropyLoss()

    def process_targets(targets, num_classes):
        targets = F.one_hot(targets, num_classes).float()
        targets = targets.permute(0, 3, 1, 2)
        return tv_tensors.Mask(targets)
    return ReMixMatch(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k, gt_labels, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, rotation_loss, process_targets)
