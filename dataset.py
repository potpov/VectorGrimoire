import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.io import read_image
from PIL import Image
import zipfile
import glob


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class MNIST(Dataset):
    """
    MNIST dataset from a root directory:

    mnist
    |
    |--------------|
    training    testing
    |              |
    0-9           0-9
    """

    def __init__(self, root, train=True, transform=None):
        super(MNIST, self)
        self.root = root
        self.train = train
        self.transform = transform

        self.image_folder = os.path.join(root, "training" if train else "testing")

        self.image_paths = []
        self.labels = []

        for label in range(10):
            label_folder = os.path.join(self.image_folder, str(label))
            image_files = os.listdir(label_folder)
            for image_file in image_files:
                image_path = os.path.join(label_folder, image_file)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)


class MNISTpp(Dataset):
    """
    MNISTpp dataset from a root directory. There are no labels available.

    mnistpp
    |
    |--------------|
    training    testing
    """

    def __init__(self, root, train=True, transform=None):
        super(MNISTpp, self)
        self.root = root
        self.train = train
        self.transform = transform

        self.image_folder = os.path.join(root, "training" if train else "testing")

        self.image_paths = []

        image_files = os.listdir(self.image_folder)
        for image_file in image_files:
            image_path = os.path.join(self.image_folder, image_file)
            self.image_paths.append(image_path)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)
    
class Emoji(Dataset):
    """
    Emoji dataset from a root directory. There are no labels available.

    emoji
    |
    |--------------|
    training    testing
    """

    def __init__(self, root, train=True, transform=None):
        super(Emoji, self)
        self.root = root
        self.train = train
        self.transform = transform

        self.image_folder = root

        self.image_paths = glob.glob(self.image_folder+"/*.png")
        train_end_idx = int(len(self.image_paths) * 0.75)
        if(self.train):
            self.image_paths = sorted(self.image_paths)[:train_end_idx]
        else:
            self.image_paths = sorted(self.image_paths)[train_end_idx:]

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, 0

    def __len__(self):
        return len(self.image_paths)


class NounProject(Dataset):
    """
    The Noun Project dataset from a root directory. Class labels are directories. No train/test split in the folder structure.

    nounproject
     |
     |--------------|--------------|--------------|
     airplane    basketball       ...           zebra
     |              |              |              |
    *.png         *.png          *.png          *.png
    """

    def __init__(self, root, train=True, transform=None):
        super(NounProject, self)
        self.root = root
        self.train = train
        self.transform = transform

        self.image_folder = root
        self.threshold = 128

        self.image_paths = []
        self.labels = []

        for i, label in enumerate(os.listdir(self.image_folder)):
            image_paths = sorted(
                glob.glob(os.path.join(self.image_folder, label) + "/*.png")
            )
            train_split_idx = int(len(image_paths) * 0.75)
            if train:
                split_image_paths = image_paths[:train_split_idx]
            else:
                split_image_paths = image_paths[train_split_idx:]
            for image_path in split_image_paths:
                self.image_paths.append(image_path)
                self.labels.append(i)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path)
        if(image.mode == "RGBA"):
            bg = Image.new("RGB", image.size, (255,255,255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        if self.transform is not None:
            image = self.transform(image)
            # binarized_image = image > self.threshold
            # binarized_image = binarized_image.float()

        return image, label

    def __len__(self):
        return len(self.image_paths)


class MNISTDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        # =========================  MNIST Dataset  =========================

        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = MNIST(
            self.data_dir,
            train=True,
            transform=train_transforms,
        )

        self.val_dataset = MNIST(
            self.data_dir,
            train=False,
            transform=val_transforms,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


class MNISTppDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        # =========================  MNIST Dataset  =========================

        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = MNISTpp(
            self.data_dir,
            train=True,
            transform=train_transforms,
        )

        self.val_dataset = MNISTpp(
            self.data_dir,
            train=False,
            transform=val_transforms,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

class EmojiDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = Emoji(
            self.data_dir,
            train=True,
            transform=train_transforms,
        )

        self.val_dataset = Emoji(
            self.data_dir,
            train=False,
            transform=val_transforms,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


class NounProjectDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10.0, fill=256),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = NounProject(
            self.data_dir,
            train=True,
            transform=train_transforms,
        )

        self.val_dataset = NounProject(
            self.data_dir,
            train=False,
            transform=val_transforms,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
