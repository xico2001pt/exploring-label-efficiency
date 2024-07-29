# Exploring Label Efficiency with Semi-Supervision and Self-Supervision Methods

This repository is a software framework designed to leverage both Semi-Supervised and Self-Supervised Learning techniques to utilize unlabeled data effectively during the training process. These methods help improve model performance by generating pseudo-labels or creating artificial labels from the data itself, enabling the model to learn useful representations. The framework is built to support a wide range of applications and tasks, providing detailed documentation for projects that aim to incorporate these advanced learning techniques.

## Table Of Contents

- [License](#license)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please **ACKNOWLEDGE THE AUTHOR** if you use this repository in your project by including a link to this repository.

## Installation

The prerequisites for this project are:

- Python 3.11+
- pip

To install this project, first clone the repository:

```bash
git clone https://github.com/xico2001pt/exploring-label-efficiency
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

To adapt this project to your needs, it is recommended to read the [documentation](docs/README.md) file. This file contains a brief overview of the available documentation and links to the different sections of the documentation.

## Usage

There are 4 tool scripts in this repository, which correspond to three different learning paradigms and a test script:
- `sl_train`
- `semisl_train`
- `selfsl_train`
- `test`

To train or test a model, run the following command, where `{TOOL_SCRIPT}` is one of the tools above and `{CONFIG_PATH}` is the path for the configuration file:

```bash
python -m src.tools.{TOOL_SCRIPT} --config configs/experiments/{CONFIG_PATH}
```

For example, to use Supervised Learning to train the CIFAR-10 dataset, the following command can be used:

```bash
python -m src.tools.sl_train --config configs/experiments/sl/cifar10/wideresnet/sl_cifar10_wideresnet.yaml
```

## Project Structure

```python
exploring-label-efficiency/
├── configs/  # holds the configuration files
│   ├── configs/  # configuration files for the experiments
│   ├── datasets.yaml
│   ├── losses.yaml
│   ├── metrics.yaml
│   ├── models.yaml
│   ├── optimizers.yaml
│   ├── schedulers.yaml
│   ├── selfsl_methods.yaml
│   ├── semisl_methods.yaml
│   └── stop_conditions.yaml
├── data/  # default directory for storing input data
├── docs/  # documentation files
├── logs/  # default directory for storing logs
├── src/  # source code
│   ├── core/  # contains the core functionalities
│   ├── datasets/  # contains the datasets
│   ├── methods/  # contains the SemiSL and SelfSL methods
│   ├── models/  # contains the models
│   ├── tools/  # scripts for training, testing, etc.
│   │   ├── selfsl_train.py
│   │   ├── semisl_train.py
│   │   ├── sl_train.py
│   │   └── test.py
│   ├── trainers/  # contains the trainer classes
│   └── utils/  # utility functions
├── weights/  # default directory for storing model weights
└── requirements.txt  # project dependencies
```

## Documentation

Read the [documentation](docs/README.md) for more details about the project and all the sections mentioned above.


## Contributing

If you want to contribute to this project, please contact the author or create a pull request with a description of the feature or bug fix.

## Acknowledgments

This repository was developed by [Francisco Cerqueira](https://github.com/xico2001pt) and used [this](https://github.com/xico2001pt/pytorch-project-template) template.
