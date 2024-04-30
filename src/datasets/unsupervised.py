import torch


class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.get_input_size = dataset.get_input_size
        self.get_num_classes = dataset.get_num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0]
