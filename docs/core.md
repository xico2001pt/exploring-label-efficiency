# Core Functionalities
This section describes the management of the core functionalities in the project, such as losses, metrics, optimizers, schedulers and stopping criteria.

## Table Of Contents
- [Custom Implementations](#custom-implementations)
    - [Losses](#losses)
- [Using A Functionality](#using-a-functionality)

## Custom Implementations
Depending on the core functionality you want to implement, create (or modify if it already exists) a file in the `src/core` directory corresponding to that functionality. The following list shows the possible core functionalities and their corresponding files:

- Losses: `src/core/losses.py`
- Metrics: `src/core/metrics.py`
- Optimizers: `src/core/optimizers.py`
- Schedulers: `src/core/schedulers.py`
- Stopping Criteria: `src/core/stopping_conditions.py`

### Losses
In order for loss functions to be used in the project, they must return a dictionary containing the loss value in the key `total`. This dictionary can also contain additional information about the loss, such as the individual components of the loss.

If you also have the intention of using the loss function without returning a dictionary, you can create a boolean parameter `return_dict` in the function signature to control this behavior.

## Using A Functionality
If you want to use a core functionality in a python file, you just need to import the desired functionality from the corresponding file in the `src/core` directory.

However, if you want to use a core functionality in a configuration file, whether it is a custom implementation or a built-in one, you need to add the functionality to the corresponding list in the `classes` dictionary inside the `core/__init__.py` file.

Then, you can follow these steps:

1. Add the functionality configuration to the intended configuration file.
2. Use the functionality reference in the intended experiment configuration file.

For more information about the configuration files, check the [Configuration Files](configs.md) page.
