# Methods
This page contains documentation about the management of methods in the project.

## Table Of Contents
- [Creating A Method](#creating-a-method)
    - [Semi-Supervised Methods](#semi-supervised-methods)
    - [Self-Supervised Methods](#self-supervised-methods)
- [Using A Method](#using-a-method)

## Creating A Method

### Semi-Supervised Methods
To create a new method, follow these steps:

1. Create a new file in the `src/methods/semisl` directory. This file will contain the method implementation.
2. Implement the method class. The class should inherit from `methods.semisl.SemiSLMethod`.
3. Make sure that the method implements the method `compute_loss(self, idx, labeled, targets, unlabeled)`. The method should return a tuple `(preds, targets, loss)` where `preds` is the model predictions for the labeled samples, `targets` is the target labels for the labeled samples, and `loss` is a dictionary containing the loss values (it must at least contain the key `'total'`).
4. Add the method to the `classes` list inside the `methods/semisl/__init__.py` file.

### Self-Supervised Methods
To create a new method, follow these steps:

1. Create a new file in the `src/methods/selfsl` directory. This file will contain the method implementation.
2. Implement the method class. The class should inherit from `methods.selfsl.SelfSLMethod`.
3. Make sure that the method implements the method `compute_loss(self, idx, unlabeled)`. The method should return a tuple `(preds, targets, loss)` where `preds` is the model predictions for the unlabeled samples, `targets` is the target labels for the unlabeled samples (can be `None`), and `loss` is a dictionary containing the loss values (it must at least contain the key `'total'`). 
4. Add the method to the `classes` list inside the `methods/selfsl/__init__.py` file.

## Using A Method
To use a method in the project, follow these steps:

1. Add the method configuration to the corresponding configuration file (`config/semisl_methods.yaml` or `config/selfsl_methods.yaml`).
2. Use the method reference in the intended experiment configuration file.

For more information about the configuration files, check the [Configuration Files](configs.md) page.
