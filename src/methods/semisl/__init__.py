from .classification.pi_model import PiModelCIFAR10, PiModelSVHN
from .classification.temporal import TemporalEnsemblingCIFAR10, TemporalEnsemblingSVHN
from .segmentation.pi_model_seg import PiModelCityscapesSeg

classes = [PiModelCIFAR10, PiModelSVHN, TemporalEnsemblingCIFAR10, TemporalEnsemblingSVHN, PiModelCityscapesSeg]  # Add the method classes here
