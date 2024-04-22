from torchvision import tv_tensors
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torchvision.transforms.v2 as v2
from .mixmatch import MixMatch
from ...utils.transforms import InvariantRandAugment


class MixMatchV2(MixMatch):
    def pseudo_labelling(self, labeled, targets, unlabeled):
        unlabeled = self.labeled_transform(unlabeled)
        return super().pseudo_labelling(labeled, targets, unlabeled)


def MixMatchV2CityscapesSeg(alpha, w_max, unsupervised_weight_rampup_length, temperature, k):
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
    return MixMatchV2(alpha, w_max, unsupervised_weight_rampup_length, temperature, k, labeled_transform, unlabeled_transform, supervised_loss, unsupervised_loss, process_targets)
