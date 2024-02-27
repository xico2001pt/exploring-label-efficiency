from .pi_model import PiModelCIFAR10, PiModelSVHN, PiModelCityscapesSeg
from .temporal import TemporalEnsemblingCIFAR10, TemporalEnsemblingSVHN
from .mixmatch import MixMatchCIFAR10, MixMatchSVHN

classes = [
    PiModelCIFAR10,
    PiModelSVHN,
    PiModelCityscapesSeg,
    TemporalEnsemblingCIFAR10,
    TemporalEnsemblingSVHN,
    MixMatchCIFAR10,
    MixMatchSVHN,
]  # Add the method classes here
