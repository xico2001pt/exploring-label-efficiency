import os
import numpy as np
import torch
import random
from datetime import datetime
from torch.utils.data import random_split
from .constants import Constants as c, ROOT_DIR


def _load_model(loader, config, logger):
    model_name = config[c.Configurations.Parameters.MODEL_CONFIG_NAME]
    model, model_config = loader.load_model(model_name)
    logger.log_config(c.Configurations.Parameters.MODEL_CONFIG_NAME, model_config)
    return model


def _get_device(logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log_device(device)
    device = torch.device(device)
    return device


def _get_config_name(loader, config_path):
    try:
        config = loader.load_config_file(config_path)
        name = config["name"]

    except Exception:
        name = None

    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if name is None else name


def process_data_path(data_path):
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_DIR, data_path)
    return data_path


def set_reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_train_val_data(dataset, train_val_split):
    generator = torch.Generator().manual_seed(c.Miscellaneous.SEED)

    if train_val_split > 1:
        train_samples = train_val_split
    else:
        train_samples = int(train_val_split * len(dataset))

    val_samples = len(dataset) - train_samples
    return random_split(dataset, [train_samples, val_samples], generator=generator)


def split_train_val_test_data(data, train_val_test_split):
    generator = np.random.RandomState(c.Miscellaneous.SEED)

    if train_val_test_split[0] > 1:
        train_samples = train_val_test_split[0]
    else:
        train_samples = int(train_val_test_split[0] * len(data))

    if train_val_test_split[1] > 1:
        val_samples = train_val_test_split[1]
    else:
        val_samples = int(train_val_test_split[1] * len(data))

    ix = generator.choice(len(data), len(data), False)

    train_indices = ix[:train_samples]
    val_indices = ix[train_samples:train_samples+val_samples]
    test_indices = ix[train_samples+val_samples:]

    return [data[i] for i in train_indices], [data[i] for i in val_indices], [data[i] for i in test_indices]


def classes_mean(input, class_dim=1):
    dims = tuple(i for i in range(len(input.shape)) if i != class_dim)
    return input.mean(dim=dims)


def backbone_getter(model):
    if 'backbone' in model._modules:
        return model._modules['backbone']
    elif 'layer4' in model._modules:
        return model._modules['layer4']
    return None


def dataset_transform_filter(split, transform):
    if split in ['val', 'test']:
        return None
    return transform
