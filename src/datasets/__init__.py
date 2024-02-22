from .cifar import CIFAR10, SemiSupervisedCIFAR10
from .cityscapes import CityscapesSeg, SemiSupervisedCityscapesSeg
from .svhn import SVHN, SemiSupervisedSVHN

classes = [
    CIFAR10, SemiSupervisedCIFAR10,
    CityscapesSeg, SemiSupervisedCityscapesSeg,
    SVHN, SemiSupervisedSVHN
]  # Add the dataset classes here
