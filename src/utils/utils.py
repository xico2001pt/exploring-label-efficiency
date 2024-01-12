import torch
from datetime import datetime
from .constants import Constants as c


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