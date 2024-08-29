# NCA-Morph: Medical Image Registration with Neural Cellular Automata (BMVC 2024)

This repository represents the official PyTorch code base for our BMVC 2024 published paper **NCA-Morph: Medical Image Registration with Neural Cellular Automata**. Our code exclusively utilizes the PyTorch version of the [VoxelMorph](https://github.com/voxelmorph/voxelmorph) framework as its foundation. For more details, please refer to [our paper; at link to paper].


## Table Of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [How to get started?](#how-to-get-started)
4. [Splits and pre-trained models](#splits-and-pre-trained-models)
5. [Citations](#citations)
6. [License](#license)

## Introduction

This BMVC 2024 submission currently includes the following registration networks
* NCA-Morph
* VoxelMorph
* ViTVNet
* TransMorph
* NICE Trans

## Installation

The simplest way to install all dependencies is by using [Anaconda](https://conda.io/projects/conda/en/latest/index.html):

1. Create a Python 3.9 environment as `conda create -n <your_conda_env> python=3.9` and activate it as `conda activate  <your_conda_env>`.
2. Install CUDA and PyTorch through conda with the command specified by [PyTorch](https://pytorch.org/). The command for Linux was at the time `conda install pytorch torchvision cudatoolkit=11.3 -c pytorch`. Our code was last tested with version 1.13. Pytorch and TorchVision versions can be specified during the installation as `conda install pytorch==<X.X.X> torchvision==<X.X.X> cudatoolkit=<X.X> -c pytorch`. Note that the cudatoolkit version should be of the same major version as the CUDA version installed on the machine, e.g. when using CUDA 11.x one should install a cudatoolkit 11.x version, but not a cudatoolkit 10.x version.
3. Navigate to the project root (where `setup.py` lives).
4. Execute `pip install -r requirements.txt` to install all required packages.


## How to get started?
- Since our code base follows the VoxelMorph Framework, all models are trained in the same fashion.
- The easiest way to start is using our `train_abstract_*.py` python files. For every baseline and Continual Learning method, we provide specific `train_abstract_*.py` python files, located in the [scripts folder](https://github.com/MECLabTUDA/NCA-Morph/tree/main/scripts/examples).
- The [eval folder](https://github.com/MECLabTUDA/NCA-Morph/tree/main/eval) contains a jupyter notebooks that was used to calculate performance metrics and plots used in our submission.


## Splits and pre-trained models
- **Splits**: The train and test splits which were used during training for all methods can be found in the [misc folder](https://github.com/MECLabTUDA/NCA-Morph/tree/main/misc/list_files) of this repository.
- **Models**: Our pre-trained models from our submission can be provided by contacting the [main author](mailto:amin.ranem@tu-darmstadt.de) upon request.
- **Prototypes**: Our generated prototypes along with the preprocessed dataset can be requested [per mail](mailto:amin.ranem@tu-darmstadt.de).

For more information about NCA-Morph, please read the following paper:
```
Ranem, A., Kalkhof, J. & Mukhopadhyay, A. (2024).
NCA-Morph: Medical Image Registration with Neural Cellular Automata.
```

## Citations
If you are using NCA-Morph or our code base for your article, please cite the following paper:
```
@article{ranem2024ncamorph,
  title={NCA-Morph: Medical Image Registration with Neural Cellular Automata},
  author={Ranem, Amin and Kalkhof, John and Mukhopadhyay, Anirban}
}
```

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
