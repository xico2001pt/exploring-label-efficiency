import logging
import os
import sys
import yaml
from .constants import Constants as c


def create_logger(log_dir: str, console_output: bool = True, file_output: bool = True) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if file_output:
        file_handler = logging.FileHandler(os.path.join(log_dir, c.Logging.OUTPUT_FILENAME))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console_output:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class Logger:
    def __init__(self, log_dir: str, console_output: bool = True, file_output: bool = True):
        self.log_dir = log_dir
        self._create_log_dir()
        self.logger = create_logger(log_dir, console_output, file_output)
        self.log = dict()

    def _create_log_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def save_log(self):
        yaml.dump(self.log, open(os.path.join(self.log_dir, c.Logging.LOG_FILENAME), "w"))

    def get_log_dir(self):
        return self.log_dir

    def get_log_dir_name(self):
        return os.path.basename(self.log_dir)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def add_log_entry(self, key: str, value):
        self.log[key] = value

    def log_yaml(self, title: str, yaml_dict: dict):
        self.info(title + "\n" + yaml.dump(yaml_dict))

    def log_config(self, name: str, config: dict):
        if config is None:
            return
        title = f"Loading {name} configuration"
        self.log_yaml(title, config)
        self.add_log_entry(name, config)

    def log_device(self, device: str):
        self.info(f"Using device: {device}")
        self.add_log_entry("device", device)

    def log_time(self, task: str, time: float):
        self.info(f"{task} took {time} seconds to complete")
        self.add_log_entry("duration", time)
