import os
import time
import argparse
from torch.utils.data import DataLoader
from ..trainers.SL_trainer import SLTrainer
from ..utils.loader import Loader
from ..utils.logger import Logger
from ..utils.constants import Constants as c, ROOT_DIR
from ..utils.utils import _load_model, _get_device, _get_config_name


CONFIGS_DIR = os.path.join(ROOT_DIR, c.Configurations.CONFIGS_DIR)
LOGS_DIR = os.path.join(ROOT_DIR, c.Logging.LOGS_DIR)


def _load_train_data(loader, train_config, model, logger):
    train_dataset_name = train_config[c.Configurations.Parameters.TRAIN_DATASET_CONFIG_NAME]
    val_dataset_name = train_config[c.Configurations.Parameters.VALIDATION_DATASET_CONFIG_NAME]
    optimizer_name = train_config[c.Configurations.Parameters.OPTIMIZER_CONFIG_NAME]
    loss_name = train_config[c.Configurations.Parameters.LOSS_CONFIG_NAME]
    metrics_names = train_config[c.Configurations.Parameters.METRICS_CONFIG_NAME]
    scheduler_name = train_config[c.Configurations.Parameters.SCHEDULER_CONFIG_NAME]
    stop_condition_name = train_config[c.Configurations.Parameters.STOP_CONDITION_CONFIG_NAME]

    train_dataset, train_dataset_config = loader.load_dataset(train_dataset_name, split="train")
    logger.log_config(c.Configurations.Parameters.TRAIN_DATASET_CONFIG_NAME, train_dataset_config)

    val_dataset, val_dataset_config = loader.load_dataset(val_dataset_name, split="val")
    logger.log_config(c.Configurations.Parameters.VALIDATION_DATASET_CONFIG_NAME, val_dataset_config)

    optimizer, optimizer_config = loader.load_optimizer(optimizer_name, model)
    logger.log_config(c.Configurations.Parameters.OPTIMIZER_CONFIG_NAME, optimizer_config)

    loss, loss_config = loader.load_loss(loss_name)
    logger.log_config(c.Configurations.Parameters.LOSS_CONFIG_NAME, loss_config)

    metrics, metrics_config = loader.load_metrics(metrics_names)
    metrics_dict = {metric_name: metrics_config[metric_name] for metric_name in metrics_names}
    logger.log_config(c.Configurations.Parameters.METRICS_CONFIG_NAME, metrics_dict)

    scheduler, scheduler_config = loader.load_scheduler(scheduler_name, optimizer)
    logger.log_config(c.Configurations.Parameters.SCHEDULER_CONFIG_NAME, scheduler_config)

    stop_condition, stop_condition_config = loader.load_stop_condition(stop_condition_name)
    logger.log_config(c.Configurations.Parameters.STOP_CONDITION_CONFIG_NAME, stop_condition_config)

    hyperparameters = train_config[c.Configurations.Parameters.HYPERPARAMETERS_CONFIG_NAME]
    logger.log_config(c.Configurations.Parameters.HYPERPARAMETERS_CONFIG_NAME, hyperparameters)

    return {
        c.Configurations.Parameters.TRAIN_DATASET_CONFIG_NAME: train_dataset,
        c.Configurations.Parameters.VALIDATION_DATASET_CONFIG_NAME: val_dataset,
        c.Configurations.Parameters.OPTIMIZER_CONFIG_NAME: optimizer,
        c.Configurations.Parameters.LOSS_CONFIG_NAME: loss,
        c.Configurations.Parameters.METRICS_CONFIG_NAME: metrics,
        c.Configurations.Parameters.SCHEDULER_CONFIG_NAME: scheduler,
        c.Configurations.Parameters.STOP_CONDITION_CONFIG_NAME: stop_condition,
        c.Configurations.Parameters.HYPERPARAMETERS_CONFIG_NAME: hyperparameters,
    }


def _get_dataloaders(train_dataset, val_dataset, batch_size, num_workers):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader


def _log_train_time(start_time, end_time, logger):
    logger.log_time("Training", end_time - start_time)


def main(args):
    loader = Loader(CONFIGS_DIR)
    config_name = _get_config_name(loader, args.config)

    log_dir = os.path.join(LOGS_DIR, config_name)
    logger = Logger(log_dir, console_output=True, output_filename=c.Logging.TRAIN_OUTPUT_FILENAME)

    try:
        logger.info("Loading configuration files...")

        config = loader.load_config_file(args.config)

        model = _load_model(loader, config, logger)

        train_config = config["SL_train"]
        data = _load_train_data(loader, train_config, model, logger)
        (
            train_dataset,
            val_dataset,
            optimizer,
            loss,
            metrics,
            scheduler,
            stop_condition,
            hyperparameters,
        ) = data.values()

        epochs, num_workers, batch_size, save_freq = hyperparameters.values()

        train_loader, validation_loader = _get_dataloaders(train_dataset, val_dataset, batch_size, num_workers)

        device = _get_device(logger)

        model.to(device)

        metrics = {metric_name: metric.to(device) for metric_name, metric in metrics.items()}

        trainer = SLTrainer(model, loss, device=device, logger=logger)

        start_time = time.time()

        trainer.train(
            train_loader,
            validation_loader,
            epochs,
            optimizer,
            scheduler=scheduler,
            stop_condition=stop_condition,
            metrics=metrics,
            save_freq=save_freq,
        )

        end_time = time.time()

        trainer.save_best_model(os.path.join(ROOT_DIR, c.Trainer.WEIGHTS_DIR), logger.get_log_dir_name())

        _log_train_time(start_time, end_time, logger)

        logger.save_log(c.Logging.TRAIN_LOG_FILENAME)

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
