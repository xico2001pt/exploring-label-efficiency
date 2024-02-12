import torchvision.models.segmentation as seg

BACKBONES = {
    "mobilenet_v3_large": seg.deeplabv3_mobilenet_v3_large,
    "resnet50": seg.deeplabv3_resnet50,
    "resnet101": seg.deeplabv3_resnet101,
}


def DeepLabV3(backbone, **kwargs):
    fn = BACKBONES[backbone]
    if fn is None:
        raise ValueError(f"Invalid backbone {backbone}")
    model = fn(**kwargs)
    return model
