import os
import time
import torch
import argparse
from torch.utils.data import DataLoader

from ..trainers.trainer import Trainer
from ..utils.loader import Loader
from ..utils.logger import Logger
from ..utils.constants import Constants as c, ROOT_DIR
from ..utils.utils import _load_model, _get_device, _get_config_name, set_reproducibility
import torch.optim.swa_utils as swa_utils


CONFIGS_DIR = os.path.join(ROOT_DIR, c.Configurations.CONFIGS_DIR)
LOGS_DIR = os.path.join(ROOT_DIR, c.Logging.LOGS_DIR)
WEIGHTS_DIR = os.path.join(ROOT_DIR, c.Trainer.WEIGHTS_DIR)


def _load_test_data(loader, test_config, logger):
    dataset_name = test_config[c.Configurations.Parameters.TEST_DATASET_CONFIG_NAME]
    loss_name = test_config[c.Configurations.Parameters.LOSS_CONFIG_NAME]
    metrics_names = test_config[c.Configurations.Parameters.METRICS_CONFIG_NAME]

    dataset, dataset_config = loader.load_dataset(dataset_name, split="test")
    logger.log_config(c.Configurations.Parameters.TEST_DATASET_CONFIG_NAME, dataset_config)

    model_weights_path = test_config[c.Configurations.Parameters.MODEL_WEIGHTS_PATH_CONFIG_NAME]

    loss, loss_config = loader.load_loss(loss_name)
    logger.log_config(c.Configurations.Parameters.LOSS_CONFIG_NAME, loss_config)

    metrics, metrics_config = loader.load_metrics(metrics_names)
    metrics_dict = {metric_name: metrics_config[metric_name] for metric_name in metrics_names}
    logger.log_config(c.Configurations.Parameters.METRICS_CONFIG_NAME, metrics_dict)

    hyperparameters = test_config[c.Configurations.Parameters.HYPERPARAMETERS_CONFIG_NAME]
    logger.log_config(c.Configurations.Parameters.HYPERPARAMETERS_CONFIG_NAME, hyperparameters)

    return {
        c.Configurations.Parameters.TEST_DATASET_CONFIG_NAME: dataset,
        c.Configurations.Parameters.MODEL_WEIGHTS_PATH_CONFIG_NAME: model_weights_path,
        c.Configurations.Parameters.LOSS_CONFIG_NAME: loss,
        c.Configurations.Parameters.METRICS_CONFIG_NAME: metrics,
        c.Configurations.Parameters.HYPERPARAMETERS_CONFIG_NAME: hyperparameters,
    }


def _get_dataloader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def _load_model_weights(model, model_weights_path, logger):
    try:
        path = os.path.join(WEIGHTS_DIR, model_weights_path)
        model.load_state_dict(torch.load(path))
        logger.info("Model weights loaded successfully")

    except Exception:
        logger.error("Failed to load model weights")
        raise


def _log_test_time(start_time, end_time, logger):
    logger.log_time("Testing", end_time - start_time)


def main(args):
    set_reproducibility(c.Miscellaneous.SEED)

    loader = Loader(CONFIGS_DIR)
    config_name = _get_config_name(loader, args.config)

    log_dir = os.path.join(LOGS_DIR, config_name)
    logger = Logger(log_dir, console_output=True, output_filename=c.Logging.TEST_OUTPUT_FILENAME)

    try:
        logger.info("Loading configuration files...")

        config = loader.load_config_file(args.config)

        model = _load_model(loader, config, logger)

        test_config = config["test"]
        data = _load_test_data(loader, test_config, logger)
        (
            dataset,
            model_weights_path,
            loss,
            metrics,
            hyperparameters,
        ) = data.values()

        num_workers, batch_size, ema_decay = hyperparameters.values()

        test_loader = _get_dataloader(dataset, batch_size, num_workers)

        device = _get_device(logger)

        model.to(device)

        metrics = {metric_name: metric.to(device) for metric_name, metric in metrics.items()}

        if ema_decay:
            model = swa_utils.AveragedModel(model)

        _load_model_weights(model, model_weights_path, logger)

        trainer = Trainer(model, device, logger, loss)

        start_time = time.time()

        trainer.test(test_loader, metrics)

        end_time = time.time()

        _log_test_time(start_time, end_time, logger)

        logger.save_log(c.Logging.TEST_LOG_FILENAME)

    except Exception:
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        c.Arguments.CONFIG_FILE_ARGUMENT_NAME,
        type=str,
        default=c.Arguments.CONFIG_FILE_DEFAULT_VALUE,
        help=c.Arguments.CONFIG_FILE_HELP,
    )
    args = args.parse_args()

    main(args)
