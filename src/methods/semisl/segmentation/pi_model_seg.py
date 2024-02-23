import torchvision.transforms.v2 as v2
from torch.nn import CrossEntropyLoss, MSELoss
from ..classification.pi_model import PiModel


def PiModelCityscapesSeg(w_max, unsupervised_weight_rampup_length):
    transform = v2.Compose([
        #v2.Identity(),
        #v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        v2.RandomAutocontrast(p=0.5),
        #v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        #v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    supervised_loss = CrossEntropyLoss(reduction='mean', ignore_index=0)
    unsupervised_loss = MSELoss(reduction='mean')
    return PiModel(w_max, unsupervised_weight_rampup_length, transform, supervised_loss, unsupervised_loss)
