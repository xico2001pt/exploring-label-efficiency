import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from .fixmatch import FixMatch
from ...utils.transforms import InvariantRandAugment


class FixMatchV2(FixMatch):
    def pseudo_labelling(self, labeled, targets, unlabeled):
        unlabeled = self.labeled_transform(unlabeled)
        return super().pseudo_labelling(labeled, targets, unlabeled)


def FixMatchV2CityscapesSeg(wu, confidence):
    labeled_transform = v2.Compose([
        v2.RandomCrop((512, 512), padding=64, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
    ])
    weak_unlabeled_transform = v2.Identity()

    strong_unlabeled_transform = v2.Compose([
        InvariantRandAugment(2, 10),
    ])

    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = CrossEntropyLoss(reduction='none')  # Important not to have reduction here

    def process_targets(targets, num_classes):
        targets = F.one_hot(targets, num_classes).float()
        targets = targets.permute(0, 3, 1, 2)
        return tv_tensors.Mask(targets)
    return FixMatchV2(
        wu,
        confidence,
        labeled_transform,
        weak_unlabeled_transform,
        strong_unlabeled_transform,
        supervised_loss,
        unsupervised_loss,
        process_targets
    )


def FixMatchV2KittiSeg(wu, confidence):
    labeled_transform = v2.Compose([
        v2.Resize((int(188 * 1.05), int(621 * 1.05))),
        v2.RandomCrop((188, 621)),
        v2.RandomHorizontalFlip(),
    ])
    weak_unlabeled_transform = v2.Identity()

    strong_unlabeled_transform = v2.Compose([
        InvariantRandAugment(2, 10),
    ])

    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = CrossEntropyLoss(reduction='none')  # Important not to have reduction here

    def process_targets(targets, num_classes):
        targets = F.one_hot(targets, num_classes).float()
        targets = targets.permute(0, 3, 1, 2)
        return tv_tensors.Mask(targets)
    return FixMatchV2(
        wu,
        confidence,
        labeled_transform,
        weak_unlabeled_transform,
        strong_unlabeled_transform,
        supervised_loss,
        unsupervised_loss,
        process_targets
    )
