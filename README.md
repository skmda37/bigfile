# BigFile

```
DataLoader Speed Comparison - CIFAR10
-----------------------------------------------------------------------------------------------------------------
| Implementation                   | Time Spent on DataLoader | Comment                                           |
| -------------------------------- | ------------------------ | ------------------------------------------------- |
| RAM (Memory-Bound)               | 34.018ms                 | Fastest: but not feasible for large datasets      |
| Multiple Files (Slow Disk-Bound) | 953.250ms                | Slow: open/close one file per sample              |
| BigFile (Fast Disk-Bound)        | 226.532ms                | Fast: single binary file with merged gzip streams |
-----------------------------------------------------------------------------------------------------------------
```

<p align="center">
  <a href="https://github.com/skmda37/bigfile"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-compatible-EE4C2C" alt="PyTorch compatible"/></a>
</p>

---

**BigFile** is a Python utility for transforming large datasets into a single binary file with merged gzip streams — fully compatible with `torch.utils.data.Dataset`. Designed for datasets too large to fit into RAM that consist of millions of compressed files, BigFile offers up to **4× faster** dataloading compared to reading each entry from its own file.

---

## Why BigFile?

Training ML models on large datasets often involves two bottlenecks: datasets that don't fit into RAM, and slow I/O from reading millions of individual compressed files. BigFile solves both by merging all data entries into a **single binary file** with indexed random access, enabling fast reads without loading the full dataset into memory — as a drop-in `torch.utils.data.Dataset` replacement.

---

## Installation

Python 3.10 and newer are supported.

Clone the repository:
```bash
git clone https://github.com/skmda37/bigfile.git
cd bigfile
```

Create and activate a virtual environment:
```bash
uv venv --python 3.10
source .venv/bin/activate
```

Install the package:
```bash
uv pip install -e .
```

Finally, install a [PyTorch version](https://pytorch.org/get-started/previous-versions/) compatible with your CUDA driver.

---

## Quick Start

```python
import torch
import torchvision
import torchvision.transforms as tf
from bigfile.bigfile_builder import BigFileBuilder

# Define your transform/encoder
encoder = MyEncoder()
def Xform(x):
    with torch.no_grad():
        return encoder(x)

# Load your image dataset
preprocess = torchvision.transforms.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)

# Build the BigFile — transforms each entry and writes to a single binary
builder = BigFileBuilder(filename='data/dataset.dat', xform=Xform, kPickle=False)
builder.doit(dataset=dataset, kZip=False)

# Use the resulting BigFile as a drop-in PyTorch Dataset
dataloader = torch.utils.data.DataLoader(builder.bigfile, batch_size=64)
for x, y in dataloader:
    ...
```

For a full walkthrough including benchmarking, see [`example/cifar10_example.py`](example/cifar10_example.py).

```bash
python example/cifar10_example.py
```

---

## Repository Structure

```
src/
└── bigfile/
    ├── modelling/
    │   ├── bigfile.py          # BigFile reader: header, offsets, coefficients, labels
    │   └── bigfile_builder.py  # BigFile builder: writes transformed data to binary
example/
└── cifar10_example.py          # End-to-end benchmark with CIFAR-10
```

---

## License

This project is licensed under the [MIT License](LICENSE).