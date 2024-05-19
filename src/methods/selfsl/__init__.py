from .simclr import SimCLRCIFAR10, SimCLRSVHN, SimCLRCityscapes
#from .byol import BYOLCIFAR10
from .byol_wrapper import BYOLCIFAR10, BYOLSVHN, BYOLCityscapes

classes = [
    SimCLRCIFAR10,
    SimCLRSVHN,
    SimCLRCityscapes,
    BYOLCIFAR10,
    BYOLSVHN,
    BYOLCityscapes,
]  # Add the method classes here
