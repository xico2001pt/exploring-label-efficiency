import torch
import torchvision
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from ..utils.utils import process_data_path


class CityscapesSeg(torch.utils.data.Dataset):
    def __init__(self, root, split='train', mode='fine'):
        root = process_data_path(root)
        self.ignore_index = 0
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        # TODO: Transformations as arguments
        transform = v2.Compose([
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])

        target_transform = v2.Lambda(lambda x: tv_tensors.Mask(x, dtype=torch.long))

        self.transforms = v2.Compose([
            v2.Resize(int(512*1.05)),  # TODO: Make this an argument
            v2.RandomCrop(512),  # TODO: Make this an argument
            v2.RandomHorizontalFlip(),
        ])

        self.dataset = torchvision.datasets.Cityscapes(root, split=split, mode=mode, target_type='semantic', transform=transform, target_transform=target_transform)

    def _convert_target(self, target):
        # Copy the target tensor
        copy = target.clone()

        # Set all pixels to ignore_index
        target.fill_(self.ignore_index)

        # Set the valid classes to their respective indices
        for i, c in enumerate(self.valid_classes):
            target[copy == c] = i+1

        return target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        target = self._convert_target(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        target = torch.squeeze(target)

        return image, target

    def get_input_size(self):
        return tuple(self[0][0].shape)

    def get_num_classes(self):
        return len(self.valid_classes) + 1  # +1 for the ignore index


if __name__ == "__main__":
    dataset = CityscapesSeg(root='/data/auto/cityscapes', split='train')

    from torchvision.utils import draw_segmentation_masks, save_image

    img_ori, target = dataset[0]

    img = img_ori.type(torch.uint8)
    target = target[0]

    # Change (H,W) to (num_classes, H, W) (one-hot encoding)
    target = torch.nn.functional.one_hot(target.long(), num_classes=dataset.get_num_classes())
    target = target.permute(2, 0, 1)
    target = target.type(torch.bool)

    res = draw_segmentation_masks(img, target, alpha=0.7, colors=['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige', 'maroon', 'coral', 'grey', 'white', 'navy', 'black'][::-1])

    # Save the image
    save_image(img_ori, 'img.png')
    save_image(res.float(), 'mask.png')
