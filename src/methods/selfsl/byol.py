from .selfsl_method import SelfSLMethod
from ...core.losses import sum_losses, BYOLLoss
from ...utils.structures import EMAv2, BackboneWrapper, MultiLayerPerceptron
from ...utils.utils import backbone_getter
import copy
import torch
import torchvision.transforms as v1


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


class BYOL(SelfSLMethod):
    def __init__(self, transform, loss, projector, predictor, ema_decay):
        self.transform = transform
        self.loss = loss
        self.online_encoder = None
        self.online_projector = projector
        self.online_predictor = predictor
        self.target_encoder = None
        self.target_projector = None
        self.ema_updater = EMAv2(ema_decay)

    def set_model(self, model):
        super().set_model(model)

        backbone = backbone_getter(model)

        self.online_encoder = BackboneWrapper(backbone)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        set_requires_grad(self.target_encoder, False)
        set_requires_grad(self.target_projector, False)
        self.online_encoder.train()
        self.online_projector.train()
        self.online_predictor.train()

    def on_start_train(self, train_data):
        self.device = train_data.device
        self.optimizer = train_data.optimizer

        self.online_encoder.to(self.device)
        self.online_projector.to(self.device)
        self.online_predictor.to(self.device)
        self.target_encoder.to(self.device)
        self.target_projector.to(self.device)

        self.optimizer.add_param_group({'params': self.online_projector.parameters()})
        self.optimizer.add_param_group({'params': self.online_predictor.parameters()})

    def on_optimize_step(self):
        self.ema_updater.update_model(self.target_encoder, self.online_encoder)
        self.ema_updater.update_model(self.target_projector, self.online_projector)

    def on_end_train(self, train_data):
        # Delete param group
        self.optimizer.param_groups = self.optimizer.param_groups[:-2]

    def compute_loss(self, idx, unlabeled):
        unlabeled1, unlabeled2 = self.transform(unlabeled), self.transform(unlabeled)
        unlabeled1, unlabeled2 = unlabeled1.to(self.device, non_blocking=True), unlabeled2.to(self.device, non_blocking=True)

        unlabeled = torch.cat([unlabeled1, unlabeled2], dim=0)

        online_projections = self.online_projector(self.online_encoder(unlabeled))
        online_predictions = self.online_predictor(online_projections)

        online_predictions1, online_predictions2 = online_predictions.chunk(2, dim=0)

        with torch.no_grad():
            target_projections = self.target_projector(self.target_encoder(unlabeled)).detach()

            target_projections1, target_projections2 = target_projections.chunk(2, dim=0)

        loss = sum_losses([
            self.loss(online_predictions1, target_projections2.detach()),
            self.loss(online_predictions2, target_projections1.detach())
        ], False)

        return None, None, {"total": loss.mean()}


def BYOLCIFAR10(ema_decay, representation_size, prediction_size, projection_size, hidden_size, image_size, color_jitter_strength):
    s = color_jitter_strength
    transform = v1.Compose([
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
        v1.RandomHorizontalFlip(p=0.5),
        v1.RandomApply([v1.GaussianBlur((3, 3))], p=0.2),
        v1.RandomResizedCrop((image_size, image_size)),
    ])
    loss = BYOLLoss()
    projector = MultiLayerPerceptron(representation_size, hidden_size, projection_size)
    predictor = MultiLayerPerceptron(projection_size, hidden_size, prediction_size)
    return BYOL(transform, loss, projector, predictor, ema_decay)


def BYOLKitti(ema_decay, representation_size, prediction_size, projection_size, hidden_size, image_size, color_jitter_strength):
    s = color_jitter_strength
    h, w = image_size
    transform = v1.Compose([
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
        v1.RandomHorizontalFlip(p=0.5),
        v1.RandomApply([v1.GaussianBlur((3, 3))], p=0.2),
        v1.Resize((int(h * 1.05), int(w * 1.05))),
        v1.RandomCrop((h, w)),
    ])
    loss = BYOLLoss()
    projector = MultiLayerPerceptron(representation_size, hidden_size, projection_size)
    predictor = MultiLayerPerceptron(projection_size, hidden_size, prediction_size)
    return BYOL(transform, loss, projector, predictor, ema_decay)
