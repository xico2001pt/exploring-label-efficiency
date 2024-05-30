from .simclr import SimCLRCIFAR10, SimCLRSVHN, SimCLRCityscapes
#from .byol import BYOLCIFAR10
from .byol_wrapper import BYOLCIFAR10, BYOLSVHN, BYOLCityscapes
from .moco import MoCoCIFAR10, MoCoSVHN, MoCoCityscapes

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
]  # Add the method classes here
