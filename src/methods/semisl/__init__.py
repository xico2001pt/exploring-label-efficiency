from .pi_model import PiModelCIFAR10, PiModelSVHN, PiModelCityscapesSeg
from .temporal import TemporalEnsemblingCIFAR10, TemporalEnsemblingSVHN

classes = [PiModelCIFAR10, PiModelSVHN, TemporalEnsemblingCIFAR10, TemporalEnsemblingSVHN, PiModelCityscapesSeg]  # Add the method classes here
