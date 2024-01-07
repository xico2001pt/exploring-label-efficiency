import torchvision


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, download=True):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.49139968, 0.48215827, 0.44653124),
                (0.24703233, 0.24348505, 0.26158768)
            )
        ])
        super(CIFAR10, self).__init__(
            root=root, train=train, transform=transform, download=download
        )

# TODO: Add CIFAR100
