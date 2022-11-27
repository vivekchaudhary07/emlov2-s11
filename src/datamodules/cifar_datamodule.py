from typing import Any, Dict, Optional, Tuple

import albumentations as A
import numpy as np
import torch

from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as T


class Cifar10Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, label


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        train_val_test_split: Tuple[int, int, int] = (45_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transforms = A.Compose(
            [
                A.Rotate(limit=5, interpolation=1, border_mode=4),
                A.HorizontalFlip(),
                A.CoarseDropout(2, 8, 8, 1, 8, 8),
                A.RandomBrightnessContrast(brightness_limit=1.5, contrast_limit=0.9),
                A.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )
        self.test_transforms = A.Compose(
            [
                A.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )

        self.dims = (3, 32, 32)
        # self.num_classes = 10

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        # download
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = CIFAR10(self.hparams.data_dir, train=True)
            testset = CIFAR10(self.hparams.data_dir, train=False)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=Cifar10Dataset(self.data_train, self.train_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=Cifar10Dataset(self.data_val, self.test_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=Cifar10Dataset(self.data_test, self.test_transforms),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
