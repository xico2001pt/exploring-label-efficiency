import os
import torch
from skimage.io import imread  # TODO: ADD TO REQUIREMENTS
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2

from .semi_supervised import SemiSupervisedDataset
from.unsupervised import UnsupervisedDataset
from ..utils.utils import process_data_path, split_train_val_test_data


class KittiSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', train_val_test_split=[0.7, 0.1, 0.2], transform=None):
        root = process_data_path(root)
        self.ignore_index = 0
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        self.image_transform = v2.Compose([
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(
                (0.3568, 0.3738, 0.3765),
                (0.3206, 0.3210, 0.3233)
            )
        ])

        self.target_transform = v2.Lambda(lambda x: tv_tensors.Mask(x, dtype=torch.long))

        self.transform = transform

        self.img_dir = self.get_img_dir(root)
        self.mask_dir = self.get_mask_dir(root)
        self.images = sorted(os.listdir(self.img_dir))

        train_test_splitted_data = split_train_val_test_data(self.images, train_val_test_split)
        if split == 'train':
            self.images = train_test_splitted_data[0]
        elif split == 'val':
            self.images = train_test_splitted_data[1]
        else:
            self.images = train_test_splitted_data[2]
    
    def get_img_dir(self, root):
        return os.path.join(root, 'semantics', 'training', 'image_2')
    
    def get_mask_dir(self, root):
        return os.path.join(root, 'semantics', 'training', 'semantic')

    def _convert_target(self, target):
        copy = target.clone()
        target.fill_(self.ignore_index)
        for i, cl in enumerate(self.valid_classes):
            target[copy == cl] = i+1

        return target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = imread(os.path.join(self.img_dir, self.images[index]))
        mask = imread(os.path.join(self.mask_dir, self.images[index]), True)

        image = self.image_transform(image)
        mask = self.target_transform(mask)
        mask = self._convert_target(mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask.squeeze()

    def get_input_size(self):
        return tuple(self[0][0].shape)

    def get_num_classes(self):
        return len(self.valid_classes) + 1  # +1 for ignore index


class UnsupervisedKittiSegDataset(KittiSegDataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__(root, split, [1.0, 0, 0], transform)

    def get_img_dir(self, root):
        return os.path.join(root, 'semantic', 'testing', 'image_2')

    def __getitem__(self, index):
        image = imread(os.path.join(self.img_dir, self.images[index]))

        image = self.image_transform(image)

        if self.transform is not None:
            image = self.transform(image)

        return image


class SemiSupervisedKittiSegDataset(SemiSupervisedDataset):
    def __init__(self, root, split='labeled', train_val_test_split=[0.7, 0.1, 0.2], num_labeled=70, transform=None):
        if split == 'train':
            split = 'labeled'

        if split == 'labeled':
            dataset = KittiSegDataset(root, 'train', train_val_test_split, transform)
        else:
            dataset = UnsupervisedKittiSegDataset(root, 'train', transform)
            num_labeled = 0

        super().__init__(dataset, split, num_labeled)


def KittiSeg(root, split='train', train_val_test_split=[0.7, 0.1, 0.2]):
    transform = v2.Compose([
        v2.Resize((int(188 * 1.05), int(621 * 1.05))),
        v2.RandomCrop((188, 621)),
    ])
    return KittiSegDataset(root, split, train_val_test_split, transform)


def SemiSupervisedKittiSeg(root, split='labeled', train_val_test_split=[0.7, 0.1, 0.2], num_labeled=70):
    transform = v2.Compose([
        v2.Resize((int(188 * 1.05), int(621 * 1.05))),
        v2.RandomCrop((188, 621)),
    ])
    return SemiSupervisedKittiSegDataset(root, split, train_val_test_split, num_labeled, transform)


def UnsupervisedKittiSeg(root, split='train'):
    transform = v2.Compose([
        v2.Resize((int(188 * 1.05), int(621 * 1.05))),
        v2.RandomCrop((188, 621)),
    ])
    return UnsupervisedDataset(root, split, transform)

