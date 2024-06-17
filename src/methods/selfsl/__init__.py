from .simclr import SimCLRCIFAR10, SimCLRSVHN, SimCLRCityscapes, SimCLRKitti
#from .byol import BYOLCIFAR10  # TODO: ADD LATER
from .byol_wrapper import BYOLCIFAR10, BYOLSVHN, BYOLCityscapes
from .moco import MoCoCIFAR10, MoCoSVHN, MoCoCityscapes, MoCoKitti
from .rotation import RotationCIFAR10, RotationSVHN, RotationCityscapes

classes = [
    SimCLRCIFAR10,
    SimCLRSVHN,
    SimCLRCityscapes,
    SimCLRKitti,

    BYOLCIFAR10,
    BYOLSVHN,
    BYOLCityscapes,

    MoCoCIFAR10,
    MoCoSVHN,
    MoCoCityscapes,
    MoCoKitti,

    RotationCIFAR10,
    RotationSVHN,
    RotationCityscapes,
]  # Add the method classes here
