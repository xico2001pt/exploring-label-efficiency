from .simclr import SimCLRCIFAR10, SimCLRSVHN, SimCLRCityscapes
#from .byol import BYOLCIFAR10  # TODO: ADD LATER
from .byol_wrapper import BYOLCIFAR10, BYOLSVHN, BYOLCityscapes
from .moco import MoCoCIFAR10, MoCoSVHN, MoCoCityscapes
from .rotation import RotationCIFAR10, RotationSVHN, RotationCityscapes

classes = [
    SimCLRCIFAR10,
    SimCLRSVHN,
    SimCLRCityscapes,
    BYOLCIFAR10,
    BYOLSVHN,
    BYOLCityscapes,
    MoCoCIFAR10,
    MoCoSVHN,
    MoCoCityscapes,
    RotationCIFAR10,
    RotationSVHN,
    RotationCityscapes,
]  # Add the method classes here
