import torch
import torchvision
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from .semi_supervised import SemiSupervisedDataset
from ..utils.utils import process_data_path, split_train_val_data


class CityscapesSeg(torch.utils.data.Dataset):
    def __init__(self, root, split='train', mode='fine', train_val_split=2475):
        root = process_data_path(root)
        self.ignore_index = 0
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        # TODO: Transformations as arguments
        image_transform = v2.Compose([
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])

        target_transform = v2.Compose([
            v2.Lambda(lambda x: tv_tensors.Mask(x, dtype=torch.long)),
            v2.Lambda(lambda x: x.squeeze()),
            v2.Lambda(lambda x: tv_tensors.Mask(x)),
        ])

        self.transforms = v2.Compose([
            v2.Resize(int(512*1.05)),  # TODO: Make this an argument
            v2.RandomCrop(512),  # TODO: Make this an argument
        ])

        if split in ['train', 'train_extra']:
            self.transforms = v2.Compose([
                self.transforms,
                v2.RandomHorizontalFlip(p=0.5)
            ])

        pseudo_split = split

        if mode == 'fine':
            if split == 'test':
                pseudo_split = 'val'
            elif split == 'val':
                pseudo_split = 'train'

        self.dataset = torchvision.datasets.Cityscapes(root, split=pseudo_split, mode=mode, target_type='semantic', transform=image_transform, target_transform=target_transform)

        if mode == 'fine' and split in ['train', 'val']:
            splitted_data = split_train_val_data(self.dataset, train_val_split)
            self.dataset = splitted_data[0] if split == 'train' else splitted_data[1]

    def _convert_target(self, target):
        copy = target.clone()
        target.fill_(self.ignore_index)
        for i, cl in enumerate(self.valid_classes):
            target[copy == cl] = i+1

        return target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        target = self._convert_target(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_input_size(self):
        return tuple(self[0][0].shape)

    def get_num_classes(self):
        return len(self.valid_classes) + 1  # +1 for the ignore index


class SemiSupervisedCityscapesSeg(SemiSupervisedDataset):
    def __init__(self, root, split='labeled', train_val_split=2475, num_labeled=372):
        if split == 'train':
            split = 'labeled'

        if split == 'labeled':
            dataset = CityscapesSeg(root, split='train', mode='fine', train_val_split=train_val_split)
        else:
            dataset = CityscapesSeg(root, split='train_extra', mode='coarse', train_val_split=train_val_split)
            num_labeled = 0

        super().__init__(dataset, split, num_labeled)


if __name__ == "__main__":
    ############  DEBUGGING  ############
    dataset = CityscapesSeg(root='/data/auto/cityscapes', split='train')

    from torchvision.utils import save_image
    from ..utils.utils import set_reproducibility

    set_reproducibility(201905337)

    img, _ = dataset[0]
    save_image(img, 'cityscapes0.png')

    # Crop and resize
    img1 = v2.Compose([
        v2.RandomCrop(512, padding=50),
    ])(img)
    save_image(img1, 'cityscapes1.png')

    # Flip
    img2 = v2.RandomHorizontalFlip(p=1)(img)
    save_image(img2, 'cityscapes2.png')

    # Color distortion
    img3 = v2.ColorJitter(brightness=(0.7, 0.7), contrast=0.5, saturation=(0.9, 0.9), hue=(0.5, 0.5))(img)
    save_image(img3, 'cityscapes3.png')

    # Rotation
    img4 = v2.RandomRotation((90, 90))(img)
    save_image(img4, 'cityscapes4.png')

    # Cutout
    img5 = v2.RandomErasing(p=1, ratio=(1, 1))(img)
    save_image(img5, 'cityscapes5.png')

    # Blur
    img6 = v2.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 5.0))(img)
    save_image(img6, 'cityscapes6.png')

    # Sharpness
    img7 = v2.RandomAdjustSharpness(sharpness_factor=8, p=1)(img)
    save_image(img7, 'cityscapes7.png')
