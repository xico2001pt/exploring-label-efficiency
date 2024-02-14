import torch
from torch.utils.data import random_split
from ..utils.constants import Constants as c


class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split='labeled', num_labeled=4000):
        if split not in ['labeled', 'unlabeled']:
            raise ValueError("split must be either 'labeled' or 'unlabeled'")

        self.split = split

        generator = torch.Generator().manual_seed(c.Miscellaneous.SEED)
        splitted_data = random_split(dataset, [num_labeled, len(dataset) - num_labeled], generator=generator)

        self.dataset = splitted_data[0] if split == 'labeled' else splitted_data[1]

        self.get_input_size = dataset.get_input_size
        self.get_num_classes = dataset.get_num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.split == 'labeled':
            return self.dataset[index]
        return self.dataset[index][0]
