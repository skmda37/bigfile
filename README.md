# About
Many large datasets do not fit into RAM and consist of millions of compressed files which greatly slow down the dataloader when training ML models. This repository contains a general Python utility to transform the dataset from RAM to a single binary file with merged gzip streams. It is fully compatible with `torch.utils.data.Dataset` and is up to 4X faster than an ImageNet dataloader reading from each entry from its own file.

# Setup
Python 3.10.x and newer are supported.

1. Clone the repository via
    ```
    git clone https://github.com/skmda37/bigfile.git
    ```
1. Navigate to the root of the repo
    ```
    cd bifgile
    ```
1. Create a virtualenv in the root of the repo via
    ```
    python -m venv venv
    ```
1. Activate the virtualenv via
    ```
    source venv/bin/activate
    ```
1. Install dependecies and the project source code as a local package via
    ```
    pip install -e .
    ```
1. If you want to use the code for a PyTorch dataloader you need to a install [pytorch version](https://pytorch.org/get-started/previous-versions/) that is compatible with your CUDA driver. For instance, if you have cuda 11.8 then you can install
    ```
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
    ```
# Content
* [src/bigfile](src/bigfile) contains the code for this project
* [src/bigfile/modelling.bigfile.py](src/bigfile/modelling.bigfile.py) contains implementations of bigfile; implements reading header, offsets, coefficients, and labels from binary file
* [src/bigfile/modelling.bigfile_builder](src/bigfile/modelling.bigfile_builder.py) contains implementation of module that builds a bigfile following builder pattern; writes transformed data entries and labels to a single binary with merged gzip streams

# Example
You can test the BigFile implementation for the cifar10 dataset in [example/cifar10_example.py](example/cifar10_example.py). The code takes the cifar10 dataset and transforms it with a convolution (toy) encoder into latent feature representations. Then we compare how long it takes to loop through the dataloader of the transformed dataset with 
* our BigFile dataset implementation
* a naive dataset implementation that loads each data entry from its own `.npy` file on disk
* a dataset that is loaded fully into RAM

Running
```python
python example/cifar10_example.py
```
gives
```
| Implementation | Time Spent on DataLoader        |
| -------------- | ------------------------------- |
| BigFile        | 226.532ms                       |
| NaiveDataset   | 953.250ms                       |
| InRAMDataset   | 34.018ms                        |
```
BigFile offers up to 4X speed ups in dataloader during training over the naive approach that reads each entry from its own file. Of course if the dataset fits into RAM we are much faster. BigFile speeds up the dataloader for datasets that don't fit into RAM.
