import os
from pathlib import Path
from collections import namedtuple
from collections.abc import Iterable
from typing import Callable, List, Union, Tuple
import time
from abc import ABCMeta, abstractmethod
import shlex
import subprocess

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as tf
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from bigfile.bigfile_builder import BigFileBuilder
from bigfile.utils.timing import time_spent


DATASETNAMES = [
    'cifar10', 'cifar100'
]


def get_dataloaders(
    batch_size: int,
    num_workers: int,
    traindata: Dataset,
    valdata: Dataset
) -> Tuple[DataLoader, DataLoader]:
    trainloader = DataLoader(
        traindata,
        batch_size=batch_size,
        num_workers=num_workers
    )
    valloader = DataLoader(
        valdata,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return trainloader, valloader


def get_img_dataset(
    datasetname: str,
    train: bool,
    dataroot: str,
    preprocess: tf.transforms
) -> Iterable:
    if datasetname == 'toy':
        return [
            (torch.randn(3, 224, 224), torch.randint(0, 10, size=(1,)))
            for _ in range(10)
        ]
    elif datasetname == 'cifar10':
        return CIFAR10(
            root=dataroot,
            train=train,
            download=True,
            transform=preprocess
        )
    elif datasetname == 'cifar100':
        return CIFAR100(
            root=dataroot,
            train=train,
            download=True,
            transform=preprocess
        )
    else:
        raise NotImplementedError(
            f'Dataset {datasetname} not supported! We support '
            + (', '.join(DATASETNAMES))
        )


class AbstractTransformedDataset(Dataset, metaclass=ABCMeta):

    def __init__(
        self,
        datasetname: str,
        Xform: Callable,
        train: bool,
        dataroot: str,
        preprocess: tf.transforms
    ):
        super().__init__()
        self.datasetname = datasetname
        self.Xform = Xform  # transform/embedder such as CLIP encoder
        self.train = train
        self.dataroot = dataroot
        self.preprocess = preprocess

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: pass


class TransformedDatasetInMemory(AbstractTransformedDataset):
    """
    Takes an image dataset and computes the Xform (a transform
    such as the CLIP embedding) for each entry and saves it in VRAM
    (if using GPU) or RAM else.
    """

    def __init__(
        self,
        datasetname: str,
        Xform: Callable,
        train: bool,
        dataroot: str,
        preprocess: tf.transforms
    ):
        super().__init__(
            datasetname=datasetname,
            Xform=Xform,
            train=train,
            dataroot=dataroot,
            preprocess=preprocess
        )
        dirpath = Path(dataroot) / 'imgembeddings' \
                                 / datasetname \
                                 / ('train' if train else 'val') \
                                 / str(Xform)
        if dirpath.exists():
            print(f'Loading Xformed dataset from {dirpath / "Xformed.pt"}...')
            self.Xformed = torch.load(dirpath / 'Xformed.pt')
            self.labels = torch.load(dirpath / 'labels.pt')
        else:
            self.Xformed, self.labels = self.compute_Xformed()
            os.makedirs(dirpath, exist_ok=True)
            torch.save(self.Xformed, dirpath / 'Xformed.pt')
            torch.save(self.labels, dirpath / 'labels.pt')

    def __len__(self) -> int: return len(self.labels)

    def compute_Xformed(
        self,
        batch_size: int = 32,
    ) -> None:
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = get_img_dataset(
            self.datasetname,
            train=self.train,
            dataroot=self.dataroot,
            preprocess=self.preprocess
        )
        Xformed, labels = [], []
        dataloader = DataLoader(dataset, batch_size=batch_size)
        desc = f'Precomputing Xformed Dataset with {self.Xform}'
        with tqdm(total=len(dataloader), colour='blue', desc=desc) as pbar:
            for i, (x, y) in enumerate(dataloader):
                #x = x.to(device)
                with torch.no_grad():
                    # Get Xform
                    Xformed.append(
                        self.Xform(x).cpu()
                    )
                # Get label
                labels.append(y)
                pbar.update(1)
        Xformed = torch.cat(Xformed, dim=0)
        labels = torch.cat(labels, dim=0)
        return Xformed, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.Xformed[idx], self.labels[idx]


class TransformedDatasetMultiFiles(AbstractTransformedDataset):
    """
    Takes an image dataset and computes Xform (transform such as CLIP
    embedding) for each entry and saves it to a .npy file. The paths
    to each file is saved in __init__(). In __getitem__() we load the file
    corresponding to passed index. NOTE this is very inefficient because
    you have to open and close a file for each entry in the dataset.
    """

    def __init__(
        self,
        datasetname: str,
        Xform: Callable,
        dirout: str,
        train: bool,
        dataroot: str,
        preprocess: tf.transforms
    ):
        super().__init__(
            datasetname=datasetname,
            Xform=Xform,
            train=train,
            dataroot=dataroot,
            preprocess=preprocess
        )

        self.dirout = dirout

        # Dataset directory
        os.makedirs(dirout, exist_ok=True)

        # Save paths of files in list
        self.files = []

        # Save labels
        self.labels = []

        # Write clip embeddings of images to files in dirout
        self.write_Xform()

    def write_Xform(self) -> None:
        print('Getting image dataset...')
        dataset = get_img_dataset(
            self.datasetname,
            train=self.train,
            dataroot=dataroot,
            preprocess=self.preprocess
        )
        desc = f'Precomputing Xformed Dataset with {self.Xform}'
        with tqdm(total=len(dataset), colour='blue', desc=desc) as pbar:
            for i, (x, y) in enumerate(dataset):
                print(f'\rPrecomputing Xform {self.Xform} {i}', end='')
                # Get Xform
                with torch.no_grad():
                    z = self.Xform(x.unsqueeze(0)).squeeze(0).cpu().numpy()
                # Save Xform to file
                path = Path(self.dirout) / f'xformed{i}.npy'
                np.save(path, z)
                self.files.append(path)
                # Save labels in list
                self.labels.append(y)
                pbar.update(1)
        print(f'Saved Xform in {self.dirout}')

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.load(self.files[idx]))
        y = torch.tensor(self.labels[idx])
        return x, y


if __name__ == '__main__':

    dataroot = './data'
    Path(dataroot).mkdir(exist_ok=True)

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    encoder = nn.Sequential(nn.Conv2d(3, 6, 5))
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #encoder.to(device)

    def Xform(x):
        with torch.no_grad():
            return encoder(x)

    """
    Test looping over dataset implementation with bigfile
    """
    bigFileBuilder = BigFileBuilder(
        filename='data/testBigFile.dat',
        xform=Xform,
        kPickle=False
    )
    bigFileBuilder.doit(
        dataset=get_img_dataset(
                    datasetname='toy',
                    train=True,
                    dataroot=dataroot,
                    preprocess=preprocess
                ),
        kZip=False
    )
    dataset = bigFileBuilder.bigfile
    tic1 = time.process_time()
    for i, (x, y) in enumerate(dataset):
        pass
    time_spent(tic1, 'loop over bigfile dataset')

    """
    Tests Naive Dataset implementation loading each entry from seperate file
    """
    dataset = TransformedDatasetMultiFiles(
        datasetname='toy',
        dirout=dataroot,
        Xform=Xform,
        train=True,
        dataroot=dataroot,
        preprocess=preprocess
    )
    tic1 = time.process_time()
    for i, (x, y) in enumerate(dataset):
        pass
    time_spent(tic1, 'loop over naive dataset')

    """
    Tests Dataset loaded into RAM (this is of course much faster)
    """
    traindataset = TransformedDatasetInMemory(
        datasetname='toy',
        Xform=Xform,
        train=True,
        dataroot=dataroot,
        preprocess=preprocess
    )
    tic1 = time.process_time()
    for i, (x, y) in enumerate(traindataset):
        pass
    time_spent(tic1, 'loop over dataset saved fully in RAM')
