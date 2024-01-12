import torchvision
from ..utils.utils import process_data_path


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, download=True):
        root = process_data_path(root)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.49139968, 0.48215827, 0.44653124),
                (0.24703233, 0.24348505, 0.26158768)
            )
        ])

        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transform
            ])

        super(CIFAR10, self).__init__(
            root=root, train=train, transform=transform, download=download
        )

# TODO: Add CIFAR100
