from .cifar import CIFAR10, SemiSupervisedCIFAR10, UnsupervisedCIFAR10, LinearEvalCIFAR10, FineTuningTrainCIFAR10
from .svhn import SVHN, SemiSupervisedSVHN, UnsupervisedSVHN, LinearEvalSVHN, FineTuningTrainSVHN
from .cityscapes import CityscapesSeg, SemiSupervisedCityscapesSeg, UnsupervisedCityscapesSeg, LinearEvalCityscapesSeg, FineTuningTrainCityscapesSeg
from .kitti import KittiSeg

classes = [
    CIFAR10,
    SemiSupervisedCIFAR10,
    UnsupervisedCIFAR10,
    LinearEvalCIFAR10,
    FineTuningTrainCIFAR10,

    SVHN,
    SemiSupervisedSVHN,
    UnsupervisedSVHN,
    LinearEvalSVHN,
    FineTuningTrainSVHN,

    CityscapesSeg,
    SemiSupervisedCityscapesSeg,
    UnsupervisedCityscapesSeg,
    LinearEvalCityscapesSeg,
    FineTuningTrainCityscapesSeg,

    KittiSeg,
]  # Add the dataset classes here
