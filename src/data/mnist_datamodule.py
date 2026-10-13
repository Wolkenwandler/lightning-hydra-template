from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule:
    """Data module for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    `src.trainer.Trainer` uses these methods:

    ```python
    def prepare_data(self):
    # Called on the main process first, then on the others: download, pre-process, save to
    # disk. Do not assign state here.

    def setup(self):
    # Called on every process: load and split the data, set variables.

    def train_dataloader(self) / val_dataloader(self) / test_dataloader(self):
    # Return the dataloaders. `batch_size` is the global batch size: the trainer splits every
    # batch across processes, so it must be divisible by the number of processes.
    ```
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The global batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Do not use it to assign state (self.x = y)."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        Called on every process after `prepare_data()`; the split is done only once.
        """
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.data_dir, train=True, transform=self.transforms)
            testset = MNIST(self.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self._dataloader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self._dataloader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self._dataloader(self.data_test, shuffle=False)

    def _dataloader(self, dataset: Dataset | None, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )


if __name__ == "__main__":
    _ = MNISTDataModule()
