from .selfsl_method import SelfSLMethod
from ...core.losses import InfoNCELoss
from ...utils.structures import EMAv2, BackboneWrapper, MultiLayerPerceptron
from ...utils.utils import backbone_getter
import copy
import torch
import torch.nn.functional as F
import torchvision.transforms as v1


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


class MoCo(SelfSLMethod):
    def __init__(self, transform, loss, decoder, queue_size, projection_size, ema_decay):
        self.transform = transform
        self.loss = loss
        self.encoder_q = None
        self.encoder_k = None
        self.decoder_q = decoder
        self.decoder_k = None
        self.queue = torch.randn(queue_size, projection_size)
        self.ema_updater = EMAv2(ema_decay)

    def set_model(self, model):
        super().set_model(model)

        backbone = backbone_getter(model)

        self.encoder_q = BackboneWrapper(backbone)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.decoder_k = copy.deepcopy(self.decoder_q)
        set_requires_grad(self.encoder_k, False)
        set_requires_grad(self.decoder_k, False)
        self.encoder_q.train()
        self.decoder_q.train()

    def on_start_train(self, train_data):
        self.device = train_data.device
        self.optimizer = train_data.optimizer

        self.encoder_q.to(self.device)
        self.encoder_k.to(self.device)
        self.decoder_q.to(self.device)
        self.decoder_k.to(self.device)

        self.optimizer.add_param_group({'params': self.decoder_q.parameters()})

        self.queue = F.normalize(self.queue.to(self.device))

    def on_optimize_step(self):
        self.ema_updater.update_model(self.encoder_k, self.encoder_q)
        self.ema_updater.update_model(self.decoder_k, self.decoder_q)

    def on_end_train(self, train_data):
        # Delete param group
        self.optimizer.param_groups = self.optimizer.param_groups[:-1]
        self.queue = None

    def compute_loss(self, idx, unlabeled):
        unlabeled1, unlabeled2 = self.transform(unlabeled), self.transform(unlabeled)
        unlabeled1, unlabeled2 = unlabeled1.to(self.device, non_blocking=True), unlabeled2.to(self.device, non_blocking=True)

        query = self.encoder_q(unlabeled1)
        query = self.decoder_q(query)
        query = F.normalize(query, dim=1)

        # Shuffle BN
        idx = torch.randperm(unlabeled2.size(0), device=self.device)
        key = self.encoder_k(unlabeled2[idx])
        key = self.decoder_k(key)
        key = F.normalize(key, dim=-1)
        key = key[torch.argsort(idx)]

        loss = self.loss(query, key, self.queue)

        self.queue = torch.cat([self.queue, query], dim=0)[key.size(0):]

        return None, None, loss


def MoCoCIFAR10(queue_size, ema_decay, temperature, representation_size, projection_size, hidden_size, image_size, color_jitter_strength):
    s = color_jitter_strength
    transform = v1.Compose([
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
        v1.RandomHorizontalFlip(p=0.5),
        v1.RandomResizedCrop((image_size, image_size)),
    ])
    loss = InfoNCELoss(temperature=temperature)
    decoder = MultiLayerPerceptron(representation_size, hidden_size, projection_size)
    return MoCo(transform, loss, decoder, queue_size, projection_size, ema_decay)


def MoCoSVHN(queue_size, ema_decay, temperature, representation_size, projection_size, hidden_size, image_size, color_jitter_strength):
    s = color_jitter_strength
    transform = v1.Compose([
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
        v1.RandomResizedCrop((image_size, image_size)),
    ])
    loss = InfoNCELoss(temperature=temperature)
    decoder = MultiLayerPerceptron(representation_size, hidden_size, projection_size)
    return MoCo(transform, loss, decoder, queue_size, projection_size, ema_decay)


def MoCoCityscapes(queue_size, ema_decay, temperature, representation_size, projection_size, hidden_size, image_size, color_jitter_strength):
    s = color_jitter_strength
    transform = v1.Compose([
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
        v1.RandomHorizontalFlip(p=0.5),
        v1.RandomResizedCrop((image_size, image_size)),
    ])
    loss = InfoNCELoss(temperature=temperature)
    decoder = MultiLayerPerceptron(representation_size, hidden_size, projection_size)
    return MoCo(transform, loss, decoder, queue_size, projection_size, ema_decay)


def MoCoKitti(queue_size, ema_decay, temperature, representation_size, projection_size, hidden_size, image_size, color_jitter_strength):
    s = color_jitter_strength
    h, w = image_size
    transform = v1.Compose([
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
        v1.RandomHorizontalFlip(p=0.5),
        v1.Resize((int(h * 1.05), int(w * 1.05))),
        v1.RandomCrop((h, w)),
    ])
    loss = InfoNCELoss(temperature=temperature)
    decoder = MultiLayerPerceptron(representation_size, hidden_size, projection_size)
    return MoCo(transform, loss, decoder, queue_size, projection_size, ema_decay)
