import torch
import torchvision
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from .semi_supervised import SemiSupervisedDataset
from ..utils.utils import process_data_path, split_train_val_data


class CityscapesSeg(torch.utils.data.Dataset):
    def __init__(self, root, split='train', mode='fine', train_val_split=0.9):
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

        target_transform = v2.Lambda(lambda x: tv_tensors.Mask(x, dtype=torch.long))

        self.transforms = v2.Compose([
            v2.Resize(int(512*1.05)),  # TODO: Make this an argument
            v2.RandomCrop(512),  # TODO: Make this an argument
        ])

        if split in ['train', 'train_extra']:
            self.transforms = v2.Compose([
                self.transforms,
                v2.RandomHorizontalFlip()
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
        # Copy the target tensor
        copy = target.clone()

        # Set all pixels to ignore_index
        target.fill_(self.ignore_index)

        # Set the valid classes to their respective indices
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

        target = torch.squeeze(target)

        return image, target

    def get_input_size(self):
        return tuple(self[0][0].shape)

    def get_num_classes(self):
        return len(self.valid_classes) + 1  # +1 for the ignore index


class SemiSupervisedCityscapesSeg(SemiSupervisedDataset):
    def __init__(self, root, split='labeled', mode='fine', num_labeled=372):
        # If split is 'labeled', select num_labeled samples from the train set
        # If split is 'unlabeled', select the samples from the extra set
        pass


if __name__ == "__main__":
    dataset = CityscapesSeg(root='/data/auto/cityscapes', split='train')

    from torchvision.utils import draw_segmentation_masks, save_image
    from ..models import DeepLabV3
    from ..utils.constants import Constants as c, ROOT_DIR
    import os

    model = DeepLabV3(backbone='resnet101', num_classes=20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    model.eval()



    WEIGHTS_DIR = os.path.join(ROOT_DIR, c.Trainer.WEIGHTS_DIR)
    path = os.path.join(WEIGHTS_DIR, 'sl_cityscapes_deeplabv3.pth')
    model.load_state_dict(torch.load(path))

    img_ori, target_ori = dataset[4]

    sample = img_ori.unsqueeze(0).to(device)
    output = model(sample)

    import torchmetrics

    jac1 = torchmetrics.JaccardIndex(num_classes=20, ignore_index=0, task='multiclass', average='macro').to(device)
    jac2 = torchmetrics.JaccardIndex(num_classes=20, ignore_index=0, task='multiclass', average='micro').to(device)
    jac3 = torchmetrics.JaccardIndex(num_classes=20, ignore_index=0, task='multiclass', average='none').to(device)
    target_ori = target_ori.unsqueeze(0).to(device)
    print(output.shape, target_ori.shape)
    met1 = jac1(output, target_ori)
    met2 = jac2(output, target_ori)
    met3 = jac3(output, target_ori)
    print(met1, met2, met3)

    max_i, max_val = 0, 0
    for i in range(20):
        val = output[0][i][0,0]
        if val > max_val:
            max_i = i
            max_val = val

    print(max_i, max_val)

    output = output.argmax(1)

    print(output[0][0,0])


    print(output.shape)
    target = output.squeeze(0)
    print(target.shape)

    img = img_ori.type(torch.uint8)

    # Change (H,W) to (num_classes, H, W) (one-hot encoding)
    target = torch.nn.functional.one_hot(target.long(), num_classes=dataset.get_num_classes())
    target = target.permute(2, 0, 1)
    target = target.type(torch.bool)

    res = draw_segmentation_masks(img, target, alpha=1.0, colors=['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige', 'maroon', 'coral', 'grey', 'white', 'navy', 'black'][::-1])

    # Save the image
    save_image(img_ori, 'img.png')
    save_image(res.float(), 'mask.png')
