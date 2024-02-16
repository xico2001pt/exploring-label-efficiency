import torch
import torchvision
from .semi_supervised import SemiSupervisedDataset
from ..utils.utils import process_data_path, split_train_val_data


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

        train_or_val = split in ['train', 'val']

        if split == 'train':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transform
            ])

        self.dataset = torchvision.datasets.CIFAR10(
            root,
            train=train_or_val,
            transform=transform,
            download=True
        )

        if train_or_val:
            splitted_data = split_train_val_data(self.dataset, train_val_split)
            self.dataset = splitted_data[0] if split == 'train' else splitted_data[1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_input_size(self):
        return tuple(self[0][0].shape)

    def get_num_classes(self):
        return 10


class SemiSupervisedCIFAR10(SemiSupervisedDataset):
    def __init__(self, root, split='labeled', train_val_split=0.9, num_labeled=4000):
        dataset = CIFAR10(root, split='train', train_val_split=train_val_split)
        super().__init__(dataset, split=split, num_labeled=num_labeled)

# TODO: Add CIFAR100
