from .cifar import CIFAR10, SemiSupervisedCIFAR10, UnsupervisedCIFAR10, LinearEvalCIFAR10, FineTuningTrainCIFAR10
from .cityscapes import CityscapesSeg, SemiSupervisedCityscapesSeg, UnsupervisedCityscapesSeg, LinearEvalCityscapesSeg, FineTuningTrainCityscapesSeg
from .svhn import SVHN, SemiSupervisedSVHN, UnsupervisedSVHN, LinearEvalSVHN, FineTuningTrainSVHN

classes = [
    CIFAR10, SemiSupervisedCIFAR10, UnsupervisedCIFAR10, LinearEvalCIFAR10, FineTuningTrainCIFAR10,
    CityscapesSeg, SemiSupervisedCityscapesSeg, UnsupervisedCityscapesSeg, LinearEvalCityscapesSeg, FineTuningTrainCityscapesSeg,
    SVHN, SemiSupervisedSVHN, UnsupervisedSVHN, LinearEvalSVHN, FineTuningTrainSVHN,
]  # Add the dataset classes here
