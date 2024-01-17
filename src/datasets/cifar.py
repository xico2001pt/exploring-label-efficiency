import torchvision
import torch
from ..utils.utils import process_data_path


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
            n_train = int(train_val_split * len(self.dataset))

            if split == 'train':
                self.dataset = torch.utils.data.Subset(self.dataset, range(n_train))
            else:
                self.dataset = torch.utils.data.Subset(self.dataset, range(n_train, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class SemiSupervisedCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, split='labeled', train_val_split=0.9, num_labeled=4000):
        if split not in ['labeled', 'unlabeled']:
            raise ValueError("split must be either 'labeled' or 'unlabeled'")

        self.split = split

        train_dataset = CIFAR10(root, 'train', train_val_split)

        if split == 'labeled':
            self.dataset = torch.utils.data.Subset(train_dataset, range(num_labeled))
        else:
            self.dataset = torch.utils.data.Subset(train_dataset, range(num_labeled, len(train_dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.split == 'labeled':
            return self.dataset[index]
        return self.dataset[index][0]


# TODO: Add CIFAR100
