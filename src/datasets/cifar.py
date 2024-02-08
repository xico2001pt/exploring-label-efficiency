import torchvision
import torch
from torch.utils.data import random_split
from ..utils.utils import process_data_path
from ..utils.constants import Constants as c


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, split='train', train_val_split=0.9):
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be either 'train', 'val' or 'test'")

        root = process_data_path(root)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.49139968, 0.48215827, 0.44653124),
                (0.24703233, 0.24348505, 0.26158768)
            )
        ])

        train = split in ['train', 'val']

        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transform
            ])

        self.dataset = torchvision.datasets.CIFAR10(
            root,
            train=train,
            transform=transform,
            download=True
        )

        if train:
            generator = torch.Generator().manual_seed(c.Miscellaneous.SEED)
            splitted_data = random_split(self.dataset, [train_val_split, 1 - train_val_split], generator=generator)

            self.dataset = splitted_data[0] if split == 'train' else splitted_data[1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_input_size(self):
        return (3, 32, 32)

    def get_num_classes(self):
        return 10

# TODO: Add CIFAR100
