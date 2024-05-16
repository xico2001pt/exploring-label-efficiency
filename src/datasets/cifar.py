import torch
import torchvision
import torchvision.transforms.v2 as v2
from .semi_supervised import SemiSupervisedDataset
from .unsupervised import UnsupervisedDataset
from ..utils.utils import process_data_path, split_train_val_data, dataset_transform_filter


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', train_val_split=0.9, transform=None):
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be either 'train', 'val' or 'test'")

        root = process_data_path(root)

        transforms = [
            v2.ToTensor(),
            v2.Normalize(
                (0.49139968, 0.48215827, 0.44653124),
                (0.24703233, 0.24348505, 0.26158768)
            )
        ]

        if transform is not None:
            transforms.append(transform)

        transform = v2.Compose(transforms)

        train_or_val = split in ['train', 'val']

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


class SemiSupervisedCIFAR10Dataset(SemiSupervisedDataset):
    def __init__(self, root, split='labeled', train_val_split=0.9, num_labeled=4000, transform=None):
        dataset = CIFAR10Dataset(root, split='train', train_val_split=train_val_split, transform=transform)
        super().__init__(dataset, split=split, num_labeled=num_labeled)


class UnsupervisedCIFAR10Dataset(UnsupervisedDataset):
    def __init__(self, root, split, train_val_split=0.9, transform=None):
        dataset = CIFAR10Dataset(root, split=split, train_val_split=train_val_split, transform=transform)
        super().__init__(dataset)


def CIFAR10(root, split='train', train_val_split=0.9):
    transform = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
    ])
    transform = dataset_transform_filter(split, transform)
    return CIFAR10Dataset(root, split=split, train_val_split=train_val_split, transform=transform)


def SemiSupervisedCIFAR10(root, split='labeled', train_val_split=0.9, num_labeled=4000, force_transform=False):
    transform = None
    if force_transform:
        transform = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
        ])
    transform = dataset_transform_filter(split, transform)
    return SemiSupervisedCIFAR10Dataset(root, split=split, train_val_split=train_val_split, num_labeled=num_labeled, transform=transform)


def UnsupervisedCIFAR10(root, split, train_val_split=0.9):
    return UnsupervisedCIFAR10Dataset(root, split, train_val_split=train_val_split, transform=None)


def LinearEvalCIFAR10(root, split, train_val_split=0.9):
    transform = v2.Compose([
        v2.Resize(128),
    ])
    return CIFAR10Dataset(root, split=split, train_val_split=train_val_split, transform=transform)


def FineTuningTrainCIFAR10(root, split, train_val_split=0.9, num_labeled=4000):
    transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.Resize(128),
    ])
    return SemiSupervisedCIFAR10Dataset(root, split=split, train_val_split=train_val_split, num_labeled=num_labeled, transform=transform)
