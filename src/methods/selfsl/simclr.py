from .selfsl_method import SelfSLMethod
from ...core.losses import NTXentLoss
from ...utils.utils import backbone_getter
from torch import nn
from collections import OrderedDict
import torchvision.transforms as v1


class SimCLR(SelfSLMethod):
    def __init__(self, transform, loss, decoder_builder):
        self.transform = transform
        self.loss = loss
        self.decoder_builder = decoder_builder
        self.decoder = None

    def set_model(self, model):
        super().set_model(model)

        def on_forward(m, inputs, output):
            if isinstance(output, OrderedDict):
                output = output['out']

            if len(output.shape) > 2:
                output = output.mean([2, 3])

            if self.decoder is None:
                self.init_decoder(output.size(1))

            self.embeddings = output

        backbone = backbone_getter(model)
        self.rot_hook = backbone.register_forward_hook(on_forward)

    def on_start_train(self, train_data):
        self.device = train_data.device
        self.optimizer = train_data.optimizer

    def on_end_train(self, train_data):
        # Delete param group
        self.optimizer.param_groups = self.optimizer.param_groups[:-1]

        # Delete hook
        self.rot_hook.remove()

    def init_decoder(self, num_features):
        self.decoder = self.decoder_builder(num_features).to(self.device)
        self.optimizer.add_param_group({'params': self.decoder.parameters()})
        self.decoder.train()

    def compute_loss(self, idx, unlabeled):
        unlabeled1, unlabeled2 = self.transform(unlabeled), self.transform(unlabeled)
        unlabeled1, unlabeled2 = unlabeled1.to(self.device, non_blocking=True), unlabeled2.to(self.device, non_blocking=True)

        self.model(unlabeled1)
        representations1 = self.embeddings
        self.model(unlabeled2)
        representations2 = self.embeddings

        projections1 = self.decoder(representations1)
        projections2 = self.decoder(representations2)

        loss = self.loss(projections1, projections2)

        outputs = self.loss.get_sim()
        labels = self.loss.get_labels()

        return outputs, labels, loss


def SimCLRCIFAR10(temperature, projection_dim, color_jitter_strength):
    s = color_jitter_strength
    transform = v1.Compose([
        v1.RandomResizedCrop(224),
        v1.RandomHorizontalFlip(p=0.5),
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
    ])
    loss = NTXentLoss(temperature, return_dict=True)

    def decoder_builder(num_features):
        return nn.Sequential(
            nn.Linear(num_features, num_features, bias=False),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim, bias=False),
        )
    return SimCLR(transform, loss, decoder_builder)


def SimCLRSVHN(temperature, projection_dim, color_jitter_strength):
    s = color_jitter_strength
    transform = v1.Compose([
        v1.RandomResizedCrop(224),
        v1.RandomApply([v1.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        v1.RandomGrayscale(p=0.2),
    ])
    loss = NTXentLoss(temperature, return_dict=True)

    def decoder_builder(num_features):
        return nn.Sequential(
            nn.Linear(num_features, num_features, bias=False),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim, bias=False),
        )
    return SimCLR(transform, loss, decoder_builder)
