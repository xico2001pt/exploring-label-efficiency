from .cifar import CIFAR10, SemiSupervisedCIFAR10, UnsupervisedCIFAR10, SimCLRLinearEvalCIFAR10
from .cityscapes import CityscapesSeg, SemiSupervisedCityscapesSeg, UnsupervisedCityscapesSeg, SimCLRLinearEvalCityscapesSeg
from .svhn import SVHN, SemiSupervisedSVHN, UnsupervisedSVHN, SimCLRLinearEvalSVHN

classes = [
    CIFAR10, SemiSupervisedCIFAR10, UnsupervisedCIFAR10, SimCLRLinearEvalCIFAR10,
    CityscapesSeg, SemiSupervisedCityscapesSeg, UnsupervisedCityscapesSeg, SimCLRLinearEvalCityscapesSeg,
    SVHN, SemiSupervisedSVHN, UnsupervisedSVHN, SimCLRLinearEvalSVHN,
]  # Add the dataset classes here
