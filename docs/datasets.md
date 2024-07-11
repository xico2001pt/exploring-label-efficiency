# Datasets
This page contains documentation about the management of datasets in the project.

## Table Of Contents
- [Creating A Dataset](#creating-a-dataset)
    - [Supervised Datasets](#supervised-datasets)
    - [Semi-Supervised Datasets](#semi-supervised-datasets)
    - [Unsupervised Datasets](#unsupervised-datasets)
- [Using A Dataset](#using-a-dataset)
- [Observations](#observations)

## Creating A Dataset
To create a new dataset, follow these steps:

1. Create a new file in the `src/datasets` directory. This fill will contain the dataset implementation.
2. Implement the dataset class. The class should inherit from one of the following classes:
    - `datasets.semi_supervised.SemiSupervisedDataset` for semi-supervised datasets.
    - `datasets.unsupervised.UnsupervisedDataset` for unsupervised datasets.
    - `torch.utils.data.Dataset` for supervised datasets or custom implementations of the previous datasets.
3. Make sure that the dataset contains the following methods (only needed if it inherits from `torch.utils.data.Dataset`):
    - `__init__(self, split, ...)` to initialize the dataset.
    - `__len__(self)` to return the number of samples in the dataset.
    - `__getitem__(self, idx)` to return a sample from the dataset.
    - `get_input_size(self)` to return the input size (dimension) of the dataset. Must return a tuple in the form `(C, H, W)` or `(H, W)` for grayscale images.
    - `get_num_classes(self)` to return the number of classes in the dataset.
4. Add the dataset to the `classes` list inside the `datasets/__init__.py` file.

### Supervised Datasets
- For supervised datasets, the dataset class should inherit from `torch.utils.data.Dataset`.
- The method `__getitem__(self, idx)` must return a tuple `(sample, target)` where `sample` is the input data and `target` is the target label.
- The possible values for the `split` parameter are `train`, `val` and `test`.

### Semi-Supervised Datasets
- For semi-supervised datasets, the dataset class should inherit from `datasets.semi_supervised.SemiSupervisedDataset`, but if a custom implementation is needed, the class can inherit from `torch.utils.data.Dataset`.
- For semi-supervised datasets, an additional parameter `num_labeled` must be passed to the constructor to specify the number of labeled samples in the dataset.
- The possible values for the `split` parameter are `train`, `labeled` and `unlabeled`, where `train` is equivalent to `labeled`.
- The method `__getitem__(self, idx)` must be able to dynamically return a tuple `(sample, target)` or a tensor `sample` depending on the split.

### Unsupervised Datasets
- For unsupervised datasets, the dataset class should inherit from `datasets.unsupervised.UnsupervisedDataset`, but if a custom implementation is needed, the class can inherit from `torch.utils.data.Dataset`.
- The possible values for the `split` parameter are `train`, `val` and `test`.
- The method `__getitem__(self, idx)` must return a tensor `sample`.

## Using A Dataset
To use a dataset in the project, follow these steps:

1. Add the dataset configuration to the `config/datasets.yaml` file.
2. Use the dataset reference in the intended experiment configuration file.

For more information about the configuration files, check the [Configuration Files](configs.md) page.

## Observations
Whenever using randomness in the dataset, make sure to set the random seed to ensure reproducibility. The random seed is defined in the `utils/constants.py` file as `Constants.Miscellaneous.SEED`. The following code snippet shows an example of how to set the random seed in the dataset:

```python
generator = torch.Generator().manual_seed(Constants.Miscellaneous.SEED)
splitted_data = random_split(dataset, [num_labeled, len(dataset) - num_labeled], generator=generator)
```
