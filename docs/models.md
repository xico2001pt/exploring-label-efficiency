# Models
This page contains documentation about the management of models in the project.

## Table Of Contents
- [Creating A Model](#creating-a-model)
- [Using A Model](#using-a-model)

## Creating A Model
To create a new model, follow these steps:

1. Create a new file in the `src/models` directory. This file will contain the model implementation.
2. Implement the model class. The class should inherit from `torch.nn.Module`.
3. (Optional) If the model is intended to be used with Self-Supervised Learning, the class must have a `backbone` attribute that contains the backbone layer.

## Using A Model
To use a model in the project, follow these steps:

1. Add the model configuration to the `config/models.yaml` file.
2. Use the model reference in the intended experiment configuration file.

For more information about the configuration files, check the [Configuration Files](configs.md) page.
