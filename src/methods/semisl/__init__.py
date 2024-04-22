from .pi_model import PiModelCIFAR10, PiModelSVHN, PiModelCityscapesSeg
from .temporal import TemporalEnsemblingCIFAR10, TemporalEnsemblingSVHN
from .mixmatch import MixMatchCIFAR10, MixMatchSVHN, MixMatchCityscapesSeg
from .mixmatch_v2 import MixMatchV2CityscapesSeg
from .remixmatch import ReMixMatchCIFAR10, ReMixMatchSVHN, ReMixMatchCityscapesSeg
from .remixmatch_v2 import ReMixMatchV2CityscapesSeg
from .fixmatch import FixMatchCIFAR10, FixMatchSVHN, FixMatchCityscapesSeg
from .fixmatch_v2 import FixMatchV2CityscapesSeg

classes = [
    PiModelCIFAR10,
    PiModelSVHN,
    PiModelCityscapesSeg,
    TemporalEnsemblingCIFAR10,
    TemporalEnsemblingSVHN,
    MixMatchCIFAR10,
    MixMatchSVHN,
    MixMatchCityscapesSeg,
    MixMatchV2CityscapesSeg,
    ReMixMatchCIFAR10,
    ReMixMatchSVHN,
    ReMixMatchCityscapesSeg,
    ReMixMatchV2CityscapesSeg,
    FixMatchCIFAR10,
    FixMatchSVHN,
    FixMatchCityscapesSeg,
    FixMatchV2CityscapesSeg,
]  # Add the method classes here
