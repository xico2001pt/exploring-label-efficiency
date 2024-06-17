from .pi_model import PiModelCIFAR10, PiModelSVHN, PiModelCityscapesSeg, PiModelKittiSeg
from .temporal import TemporalEnsemblingCIFAR10, TemporalEnsemblingSVHN
from .mixmatch import MixMatchCIFAR10, MixMatchSVHN, MixMatchCityscapesSeg
from .mixmatch_v2 import MixMatchV2CityscapesSeg, MixMatchV2KittiSeg
from .remixmatch import ReMixMatchCIFAR10, ReMixMatchSVHN, ReMixMatchCityscapesSeg
from .remixmatch_v2 import ReMixMatchV2CityscapesSeg, ReMixMatchV2KittiSeg
from .fixmatch import FixMatchCIFAR10, FixMatchSVHN, FixMatchCityscapesSeg
from .fixmatch_v2 import FixMatchV2CityscapesSeg, FixMatchV2KittiSeg

classes = [
    PiModelCIFAR10,
    PiModelSVHN,
    PiModelCityscapesSeg,
    PiModelKittiSeg,

    TemporalEnsemblingCIFAR10,
    TemporalEnsemblingSVHN,

    MixMatchCIFAR10,
    MixMatchSVHN,
    MixMatchCityscapesSeg,
    MixMatchV2CityscapesSeg,
    MixMatchV2KittiSeg,

    ReMixMatchCIFAR10,
    ReMixMatchSVHN,
    ReMixMatchCityscapesSeg,
    ReMixMatchV2CityscapesSeg,
    ReMixMatchV2KittiSeg,

    FixMatchCIFAR10,
    FixMatchSVHN,
    FixMatchCityscapesSeg,
    FixMatchV2CityscapesSeg,
    FixMatchV2KittiSeg,
]  # Add the method classes here
