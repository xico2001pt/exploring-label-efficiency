import torch
import torchvision
from torch.utils.data import random_split
from ..utils.utils import process_data_path
from ..utils.constants import Constants as c


class Cityscapes(torchvision.datasets.Cityscapes):
    def __init__(self, root, split='train', mode='fine', target_type='semantic'):
        root = process_data_path(root)
        self.ignore_index = 255
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        # TODO: ADD DATA AUGMENTATION
        super().__init__(root, split=split, mode=mode, target_type=target_type)

    def __getitem__(self, index):
        # TODO: Filter ignored
        return super().__getitem__(index)

    def get_input_size(self):
        return (3, 1024, 2048)

    def get_num_classes(self):
        return len(self.valid_classes)


if __name__ == "__main__":
    dataset = Cityscapes("/data/auto/cityscapes", split='train')
    print(dataset[0][0].size)
