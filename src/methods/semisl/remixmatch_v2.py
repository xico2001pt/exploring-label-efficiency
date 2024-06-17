import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from .remixmatch import ReMixMatch
from ...core.losses import CrossEntropyWithLogitsLoss
from ...utils.transforms import InvariantRandAugment


class ReMixMatchV2(ReMixMatch):
    def compute_loss(self, idx, labeled, targets, unlabeled):
        unlabeled = self.labeled_transform(unlabeled)
        return super().compute_loss(idx, labeled, targets, unlabeled)

    def apply_distribution_alignment(self, gt_labels, preds):
        #ratio = (1e-6 + gt_labels) / (1e-6 + self.preds_moving_average.get_value())
        #preds = preds * ratio.view(-1, 1, 1)
        self.preds_moving_average.update(preds)
        return preds


def ReMixMatchV2CityscapesSeg(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k):
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
    return ReMixMatchV2(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k, gt_labels, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, rotation_loss, process_targets)


def ReMixMatchV2KittiSeg(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k):
    gt_labels = None
    labeled_transform = v2.Compose([
        v2.Resize((int(188 * 1.05), int(621 * 1.05))),
        v2.RandomCrop((188, 621)),
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
    return ReMixMatchV2(alpha, wu_max, wu1_max, wr, unsupervised_weight_rampup_length, temperature, k, gt_labels, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, rotation_loss, process_targets)
