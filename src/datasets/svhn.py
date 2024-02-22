import torch
import torchvision
import torchvision.transforms.v2 as v2
from .semi_supervised import SemiSupervisedDataset
from ..utils.utils import process_data_path, split_train_val_data


class SVHN(torch.utils.data.Dataset):
    def __init__(self, root, split='train', train_val_split=0.9):
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be either 'train', 'val' or 'test'")

        root = process_data_path(root)

        transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(
                (0.4377, 0.4438, 0.4728),
                (0.1980, 0.2010, 0.1970)
            )
        ])

        train_or_val = split in ['train', 'val']

        self.dataset = torchvision.datasets.SVHN(
            root,
            split="train" if train_or_val else "test",
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


class SemiSupervisedSVHN(SemiSupervisedDataset):
    def __init__(self, root, split='labeled', train_val_split=0.9, num_labeled=1000):
        dataset = SVHN(root, split='train', train_val_split=train_val_split)
        super().__init__(dataset, split=split, num_labeled=num_labeled)
