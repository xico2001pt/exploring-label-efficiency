import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from .semisl_method import SemiSLMethod


def default_process_targets(targets, num_classes):
    return F.one_hot(targets, num_classes).float()


class FixMatch(SemiSLMethod):
    def __init__(self, wu, confidence, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, process_targets=default_process_targets):
        self.wu = wu
        self.confidence = confidence
        self.labeled_transform = labeled_transform
        self.weak_unlabeled_transform = weak_unlabeled_transform
        self.strong_unlabeled_transform = strong_unlabeled_transform
        self.supervised_loss = supervised_loss
        self.unsupervised_loss = unsupervised_loss
        if process_targets is None:
            process_targets = default_process_targets
        self.process_targets = process_targets

    def on_start_train(self, train_data):
        self.num_classes = train_data.num_classes

    def compute_loss(self, idx, labeled, targets, unlabeled):
        labeled, targets, strong_unlabeled, preds = self.pseudo_labelling(labeled, targets, unlabeled)

        inputs = torch.cat([labeled, strong_unlabeled], dim=0)
        outputs = self.model(inputs)
        del inputs

        unlabeled_idx = labeled.size(0)
        labeled_outputs = outputs[:unlabeled_idx]
        supervised_loss = self.supervised_loss(labeled_outputs, targets)

        strong_unlabeled_outputs = outputs[unlabeled_idx:]
        preds = torch.cat([labeled_outputs, preds], dim=0)

        # Apply threshold to the predictions
        preds = preds.detach().softmax(dim=1)
        max_probs, preds = preds.max(dim=1)
        mask = max_probs.ge(self.confidence).float()

        unsupervised_loss = self.unsupervised_loss(strong_unlabeled_outputs, preds) * mask
        unsupervised_loss = unsupervised_loss.mean()
        unsupervised_weighted_loss = unsupervised_loss * self.wu

        total_loss = supervised_loss + unsupervised_weighted_loss

        loss = {
            'total': total_loss,
            'supervised': supervised_loss,
            'unsupervised': unsupervised_loss,
            'unsupervised_weighted': unsupervised_weighted_loss,
        }

        return labeled_outputs, targets.argmax(dim=1), loss

    def pseudo_labelling(self, labeled, targets, unlabeled):
        targets = self.process_targets(targets, self.num_classes)
        labeled, targets = self.labeled_transform(labeled, targets)

        with torch.no_grad():
            weak_unlabeled = self.weak_unlabeled_transform(unlabeled)
            preds = self.model(weak_unlabeled)

        unlabeled = torch.cat([labeled, unlabeled], dim=0)
        strong_unlabeled = self.strong_unlabeled_transform(unlabeled)

        return labeled, targets, strong_unlabeled, preds


def FixMatchCIFAR10(wu, confidence):
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
    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = CrossEntropyLoss(reduction='none')  # Important not to have reduction here
    return FixMatch(wu, confidence, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss)


def FixMatchSVHN(wu, confidence):
    labeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    ])
    weak_unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    ])
    strong_unlabeled_transform = v2.Compose([
        v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
        v2.RandAugment(2, 10),
        v2.RandomErasing(scale=(0.1, 0.3), ratio=(1, 1), value=0.5, p=1.0),
    ])
    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = CrossEntropyLoss(reduction='none')  # Important not to have reduction here
    return FixMatch(wu, confidence, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss)


def FixMatchCityscapesSeg(wu, confidence):
    labeled_transform = v2.Compose([
        v2.RandomCrop((512, 512), padding=64, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
    ])
    weak_unlabeled_transform = v2.Identity()

    strong_unlabeled_transform = v2.Compose([
        v2.ColorJitter(brightness=0.5),
        v2.RandomAutocontrast(p=0.5),
        v2.RandomEqualize(p=0.5),
        v2.RandomErasing(scale=(0.1, 0.1), ratio=(1, 1), value=0.5, p=1.0),
    ])

    supervised_loss = CrossEntropyLoss()
    unsupervised_loss = CrossEntropyLoss(reduction='none')  # Important not to have reduction here

    def process_targets(targets, num_classes):
        targets = F.one_hot(targets, num_classes).float()
        targets = targets.permute(0, 3, 1, 2)
        return tv_tensors.Mask(targets)
    return FixMatch(wu, confidence, labeled_transform, weak_unlabeled_transform, strong_unlabeled_transform, supervised_loss, unsupervised_loss, process_targets)
