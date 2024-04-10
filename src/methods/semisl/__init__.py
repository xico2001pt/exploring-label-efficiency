from .pi_model import PiModelCIFAR10, PiModelSVHN, PiModelCityscapesSeg
from .temporal import TemporalEnsemblingCIFAR10, TemporalEnsemblingSVHN
from .mixmatch import MixMatchCIFAR10, MixMatchSVHN, MixMatchCityscapesSeg
from .remixmatch import ReMixMatchCIFAR10, ReMixMatchSVHN, ReMixMatchCityscapesSeg
from .fixmatch import FixMatchCIFAR10, FixMatchSVHN, FixMatchCityscapesSeg

classes = [
    PiModelCIFAR10,
    PiModelSVHN,
    PiModelCityscapesSeg,
    TemporalEnsemblingCIFAR10,
    TemporalEnsemblingSVHN,
    MixMatchCIFAR10,
    MixMatchSVHN,
    MixMatchCityscapesSeg,
    ReMixMatchCIFAR10,
    ReMixMatchSVHN,
    ReMixMatchCityscapesSeg,
    FixMatchCIFAR10,
    FixMatchSVHN,
    FixMatchCityscapesSeg,
]  # Add the method classes here
