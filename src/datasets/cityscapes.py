import torch
import torchvision
from torch.utils.data import random_split
from ..utils.utils import process_data_path
from ..utils.constants import Constants as c


class CityscapesSeg(torchvision.datasets.Cityscapes):
    def __init__(self, root, split='train', mode='fine'):
        root = process_data_path(root)
        self.ignore_index = 255
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        # TODO: Transformations as arguments
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])

        target_transform = torchvision.transforms.ToTensor()

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(512),  # TODO: Make this an argument
        ])

        super().__init__(root, split=split, mode=mode, target_type='semantic', transform=transform, target_transform=target_transform, transforms=transforms)

    def _convert_target(self, target):
        mask = torch.ones_like(target) * self.ignore_index
        for i, cl in enumerate(self.valid_classes):
            mask[target == cl] = i
        return mask

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        target = self._convert_target(target)
        return image, target

    def get_input_size(self):
        return tuple(self[0][0].shape)

    def get_num_classes(self):
        return len(self.valid_classes)
