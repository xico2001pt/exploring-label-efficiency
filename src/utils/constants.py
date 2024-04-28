import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Root directory of the project


class Constants:
    class Arguments:  # Command line arguments constants
        CONFIG_FILE_ARGUMENT_NAME = "--config"  # Name of the configuration file argument
        CONFIG_FILE_DEFAULT_VALUE = "config.yaml"  # Default value of the configuration file argument
        CONFIG_FILE_HELP = "Path to the configuration file"  # Help message of the configuration file argument

    class Configurations:  # Configuration file constants
        CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")  # Root directory where all configurations are stored

        class Parameters:  # Configuration file parameters constants
            MODEL_CONFIG_NAME = "model"  # Name of the model configuration parameter
            TRAIN_DATASET_CONFIG_NAME = "train_dataset"  # Name of the train dataset configuration parameter
            VALIDATION_DATASET_CONFIG_NAME = "val_dataset"  # Name of the validation dataset configuration parameter
            TEST_DATASET_CONFIG_NAME = "test_dataset"  # Name of the test dataset configuration parameter
            OPTIMIZER_CONFIG_NAME = "optimizer"  # Name of the optimizer configuration parameter
            METHOD_CONFIG_NAME = "method"  # Name of the semi/self-supervised method configuration parameter
            LOSS_CONFIG_NAME = "loss"  # Name of the loss configuration parameter
            VAL_LOSS_CONFIG_NAME = "val_loss"  # Name of the validation loss configuration parameter
            METRICS_CONFIG_NAME = "metrics"  # Name of the metrics configuration parameter
            SCHEDULER_CONFIG_NAME = "scheduler"  # Name of the scheduler configuration parameter
            STOP_CONDITION_CONFIG_NAME = "stop_condition"  # Name of the stop condition configuration parameter
            MODEL_WEIGHTS_PATH_CONFIG_NAME = "model_weights_path"  # Name of the model weights path configuration parameter
            MODEL_WEIGHTS_NAME_CONFIG_NAME = "model_weights_name"  # Name of the model weights in SelfSL configuration parameter
            HYPERPARAMETERS_CONFIG_NAME = "hyperparameters"  # Name of the hyperparameters configuration parameter

            EPOCHS_CONFIG_NAME = "epochs"  # Name of the epochs configuration parameter
            NUM_WORKERS_CONFIG_NAME = "num_workers"  # Name of the number of workers configuration parameter
            BATCH_SIZE_CONFIG_NAME = "batch_size"  # Name of the batch size configuration parameter
            BATCH_SIZE_LABELED_CONFIG_NAME = "batch_size_labeled"  # Name of the labeled batch size configuration parameter
            BATCH_SIZE_UNLABELED_CONFIG_NAME = "batch_size_unlabeled"  # Name of the unlabeled batch size configuration parameter

    class Loader:  # Loader constants
        DATASETS_CONFIG_FILENAME = "datasets.yaml"  # Name of the datasets configuration file
        MODELS_CONFIG_FILENAME = "models.yaml"  # Name of the models configuration file
        LOSSES_CONFIG_FILENAME = "losses.yaml"  # Name of the losses configuration file
        METRICS_CONFIG_FILENAME = "metrics.yaml"  # Name of the metrics configuration file
        OPTIMIZERS_CONFIG_FILENAME = "optimizers.yaml"  # Name of the optimizers configuration file
        SCHEDULERS_CONFIG_FILENAME = "schedulers.yaml"  # Name of the schedulers configuration file
        STOP_CONDITIONS_CONFIG_FILENAME = "stop_conditions.yaml"  # Name of the stop conditions configuration file
        SEMISL_METHODS_CONFIG_FILENAME = "semisl_methods.yaml"  # Name of the semi-supervised methods configuration file
        SELFSL_METHODS_CONFIG_FILENAME = "selfsl_methods.yaml"  # Name of the self-supervised methods configuration file

    class Logging:  # Logging constants
        LOGS_DIR = os.path.join(ROOT_DIR, "logs")  # Root directory where all logs are stored
        TRAIN_OUTPUT_FILENAME = "train_output.log"  # Name of the file that contains the redirected train output of the console
        TEST_OUTPUT_FILENAME = "test_output.log"  # Name of the file that contains the redirected test output of the console
        TRAIN_LOG_FILENAME = "train_log.yaml"  # Name of the file that contains the train configurations of the run
        TEST_LOG_FILENAME = "test_log.yaml"  # Name of the file that contains the test configurations of the run

    class Trainer:  # Trainer constants
        WEIGHTS_DIR = "weights"  # Root directory where model weights are stored

        class Checkpoints:  # Checkpoints constants
            CHECKPOINTS_DIR = "checkpoints"  # Directory (inside the run log directory) where checkpoints are stored
            BEST_CHECKPOINT_FILENAME = "best_checkpoint.pth"  # Name of the best checkpoint file
            LATEST_CHECKPOINT_FILENAME = "latest_checkpoint.pth"  # Name of the latest checkpoint file

    class Miscellaneous:
        SEED = 201905337  # Seed for reproducibility
