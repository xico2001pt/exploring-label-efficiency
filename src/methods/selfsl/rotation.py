from .selfsl_method import SelfSLMethod
from ...core.losses import CrossEntropyLoss
from ...utils.structures import BackboneWrapper
from ...utils.utils import backbone_getter
import torch
import torchvision.transforms as v1


class Rotation(SelfSLMethod):
    def __init__(self, transform, loss, representation_size):
        self.transform = transform
        self.loss = loss
        self.representation_size = representation_size
        self.encoder = None
        self.decoder = None

    def set_model(self, model):
        super().set_model(model)

        backbone = backbone_getter(model)

        self.encoder = BackboneWrapper(backbone)
        self.decoder = torch.nn.Linear(self.representation_size, 4)
        self.encoder.train()
        self.decoder.train()

    def on_start_train(self, train_data):
        self.device = train_data.device
        self.optimizer = train_data.optimizer

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.optimizer.add_param_group({'params': self.decoder.parameters()})

    def on_end_train(self, train_data):
        # Delete param group
        self.optimizer.param_groups = self.optimizer.param_groups[:-1]

    def compute_loss(self, idx, unlabeled):
        x = self.transform(unlabeled)

        x, targets = self.rotate_batch(x)

        preds = self.encoder(x)
        preds = self.decoder(preds)

        loss = self.loss(preds, targets)

        return preds, targets, loss

    def rotate_batch(self, x):
        rotations_1 = torch.rot90(x, 1, [2, 3])
        rotations_2 = torch.rot90(x, 2, [2, 3])
        rotations_3 = torch.rot90(x, 3, [2, 3])
        targets = torch.zeros(x.size(0), dtype=torch.long)

        x = torch.cat([x, rotations_1, rotations_2, rotations_3], dim=0)
        targets = torch.cat([targets, targets + 1, targets + 2, targets + 3], dim=0)
        targets = targets.to(x.device)

        return x, targets


def RotationCIFAR10(representation_size, image_size, color_jitter_strength):
    s = color_jitter_strength
    transform = v1.Compose([
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
        v1.RandomHorizontalFlip(p=0.5),
        v1.RandomResizedCrop((image_size, image_size)),
    ])
    loss = CrossEntropyLoss()
    return Rotation(transform, loss, representation_size)


def RotationSVHN(representation_size, image_size, color_jitter_strength):
    s = color_jitter_strength
    transform = v1.Compose([
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
        v1.RandomResizedCrop((image_size, image_size)),
    ])
    loss = CrossEntropyLoss()
    return Rotation(transform, loss, representation_size)


def RotationCityscapes(representation_size, image_size, color_jitter_strength):
    s = color_jitter_strength
    transform = v1.Compose([
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
        v1.RandomHorizontalFlip(p=0.5),
        v1.RandomResizedCrop((image_size, image_size)),
    ])
    loss = CrossEntropyLoss()
    return Rotation(transform, loss, representation_size)
