from ..utils.logger import Logger


class TrainData:
    def __init__(self):
        self.logger: Logger = None
        self.device = None
        self.input_size: tuple = (0, 0)
        self.num_classes: int = 0
        self.dataset_size: dict = {}
        self.batches_per_epoch: int = 0
