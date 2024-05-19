from byol_pytorch import BYOL
from .selfsl_method import SelfSLMethod
from ...utils.utils import backbone_getter
from ...utils.structures import BackboneWrapper
import torchvision.transforms as v1


class BYOLWrapper(SelfSLMethod):
    def __init__(self, transform, image_size, projection_size, hidden_size, ema_decay):
        self.transform = transform
        self.image_size = image_size
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.ema_decay = ema_decay

    def set_model(self, model):
        super().set_model(model)

        backbone = backbone_getter(model)
        backbone = BackboneWrapper(backbone)

        self.method = BYOL(
            backbone,
            self.image_size,
            hidden_layer=-1,
            projection_size=self.projection_size,
            projection_hidden_size=self.hidden_size,
            augment_fn=self.transform,
            augment_fn2=self.transform,
            moving_average_decay=self.ema_decay,
            use_momentum=True,
        )

    def on_start_train(self, train_data):
        self.device = train_data.device
        self.optimizer = train_data.optimizer

        self.method.to(self.device)
        self.optimizer.param_groups.clear()
        self.optimizer.state.clear()
        self.optimizer.add_param_group({'params': self.method.parameters()})

    def on_optimize_step(self):
        self.method.update_moving_average()

    def compute_loss(self, idx, unlabeled):
        unlabeled = v1.Resize(self.image_size)(unlabeled)
        loss = self.method(unlabeled)

        return None, None, {"total": loss}


def BYOLCIFAR10(ema_decay, representation_size, prediction_size, projection_size, hidden_size, image_size, color_jitter_strength):
    transform = None
    return BYOLWrapper(transform, image_size, projection_size, hidden_size, ema_decay)


def BYOLSVHN(ema_decay, representation_size, prediction_size, projection_size, hidden_size, image_size, color_jitter_strength):
    transform = None
    return BYOLWrapper(transform, image_size, projection_size, hidden_size, ema_decay)


def BYOLCityscapes(ema_decay, representation_size, prediction_size, projection_size, hidden_size, image_size, color_jitter_strength):
    transform = None
    return BYOLWrapper(transform, image_size, projection_size, hidden_size, ema_decay)
