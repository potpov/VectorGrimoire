import os
import torch
from torch import Tensor
from typing import List, Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import pandas as pd
import numpy as np
import string
from thesis.utils import svg2paths2, disvg, raster, get_single_paths, get_similar_length_paths, check_for_continouity, get_rasterized_segments, all_paths_to_max_diff, Path, svg_string_to_tensor
import copy
import random
import math

class VQDataset(Dataset):
    def __init__(self, csv_path:str, context_length: int,train:bool = True):
        self.split = pd.read_csv(csv_path)
        self.train = train
        self.context_length = context_length
        if self.train:
            self.vq_data: np.ndarray = np.load(self.split[self.split["split"] == "train"]["vq_token_path"].iloc[0])
            self.text_data: np.ndarray = np.load(self.split[self.split["split"] == "train"]["text_token_path"].iloc[0])
        else:
            self.vq_data: np.ndarray = np.load(self.split[self.split["split"] == "test"]["vq_token_path"].iloc[0])
            self.text_data: np.ndarray = np.load(self.split[self.split["split"] == "test"]["text_token_path"].iloc[0])
        print(f"Loaded datasets. \nVQ data shape: {self.vq_data.shape} and dtype: {self.vq_data.dtype}\nText data shape: {self.text_data.shape} and dtype: {self.text_data.dtype}")
        print("Processing...")

        cls_token = 101
        sos_token = 0
        bos_token = 1
        eos_token = 2
        pad_token = 3

        self.text_data = np.split(self.text_data, np.where(self.text_data == cls_token)[0])[1:]  # as 101 is the <CLS> token

        self.vq_data = np.split(self.vq_data, np.where(self.vq_data == bos_token)[0])[1:]  # as 1 is the <BOS> token
        self.vq_data = [x for x in self.vq_data if len(x) < self.context_length - 1 - 15]  # -1 so we can shift one position for the target, -15 because this is the max of text token amount
        assert len(self.vq_data) == len(self.text_data), "VQ and text data should have the same length."
        self.data = [np.append(x, y) for x, y in zip(self.vq_data, self.text_data)]
        # now add the SOS at index=0 and EOS token at last index to each sequence
        self.data = [np.append(np.insert(array, 0, sos_token), eos_token) for array in self.data]

        for i, array in enumerate(self.data):
            if len(array) < self.context_length:
                self.data[i] = np.append(array, np.zeros(self.context_length - len(array), dtype=np.ushort) + pad_token)
        self.data = np.stack(self.data)
        print("Finished processing dataset.")
        print(f"Dataset now with shape {self.data.shape} and dtype: {self.data.dtype}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx:int):
        row = self.data[idx]
        # TODO this might need to be changed so the targets are SVG only and inputs are SVG + text
        inputs = torch.from_numpy(row.astype(np.int32)).long()
        targets = torch.from_numpy(np.roll(row, -1).astype(np.int32)).long()
        targets[-1] = 2  # <PAD> token
        return inputs, targets
    
class VQDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        context_length: int,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.csv_path = csv_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.context_length = context_length

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = VQDataset(
            self.csv_path,
            context_length=self.context_length,
            train=True,
        )

        self.val_dataset = VQDataset(
            self.csv_path,
            context_length=self.context_length,
            train=False,
        )

    #       ===============================================================

    def collate_fn(self, batch):
        # sequences = zip(*batch)
        sequences = torch.concat(batch)
        return sequences

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            # collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            # collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            # collate_fn=self.collate_fn
        )
    
class GlyphazznStage1Dataset(Dataset):
    """
    Glyphazzn dataset that requires already normalized SVGs. Yields patches, positions, and labels. The label is the index of string.printable -> label = string.printable.index(label)

    Requires the following structure:
    top_level_dir
    |________________________________________
    |                   |                   |
    train               test                split.csv (with columns: file_path, class, split, description)
    |                   |
    0-9, a-z, A-Z       0-9, a-z, A-Z
    |                   |
    *.svg               *.svg

    Args:
        - top_level_dir: path to the top level directory of all SVGs
        - channels: number of channels for the rasterized images
        - width: width/height of the rasterized images
        - train: whether to use the train or test split
        - subset: "all", "numbers", "letters", "lowercase", or "uppercase"
        - individual_min_length: minimum length of a path segment to qualify for being a single shape layer
        - individual_max_length: maximum length of a path segment, everything longer than this will be cropped into multiple segments
        - stroke_width: stroke width for rasterization
        - max_shapes_per_svg: maximum number of shape layers per svg file, can be tuned for VRAM savings
    """

    def __init__(self,
                 top_level_dir: str,
                 channels: int,
                 width: int,
                 train: bool = True,
                 subset:str = "all",
                 individual_min_length: float = 1.,
                 individual_max_length: float = 10.,
                 stroke_width: float = 0.3,
                 max_shapes_per_svg: int = 64,
                 **kwargs):
        super(GlyphazznStage1Dataset, self)
        print(f"[INFO] These keywords were provided but are not used: {kwargs.keys()}")
        self.top_level_dir = top_level_dir
        self.individual_min_length = individual_min_length
        self.individual_max_length = individual_max_length
        self.stroke_width = stroke_width
        self.max_shapes_per_svg = max_shapes_per_svg
        self.channels = channels
        self.width = width
        self.train = train
        self.subset = subset
        df = pd.read_csv(os.path.join(top_level_dir, "split.csv"))
        self.df = df[df["split"] == ("train" if self.train else "test")].reset_index(drop=True)

        
        if train is not None:
            self.split = glob.glob(os.path.join(top_level_dir, f"{'train' if self.train else 'test'}/**/*.svg"), recursive=True)
        else:
            self.split = glob.glob(os.path.join(top_level_dir, "**/*.svg"), recursive=True)
            print("[WARNING] Using the whole dataset! Train was None.")
        if self.subset == "all":
            self.split = self.split
        elif self.subset == "numbers":
            self.split = [x for x in self.split if x.split("/")[-2] in [str(i) for i in range(10)]]
        elif self.subset == "letters":
            self.split = [x for x in self.split if x.split("/")[-2] in string.ascii_letters]
        elif self.subset == "lowercase":
            self.split = [x for x in self.split if x.split("/")[-2] in string.ascii_lowercase]
        elif self.subset == "uppercase":
            self.split = [x for x in self.split if x.split("/")[-2] in string.ascii_uppercase]
        else:
            raise ValueError(f"Subset {self.subset} not recognized.")

    def crop_path_into_segments(self, path:Path, length:float = 5.):
        """
        a single input path is cropped into segments of approx length `length`. I say "approx" because we divide the path into same length segments, which will not be exactly `length` long.
        """
        segments = []
        num_iters = math.ceil(path.length() / length)
        for i in range(num_iters):
            cropped_segment = path.cropped(i/num_iters, (i+1)/num_iters)
            segments.append(cropped_segment)

        return segments

    def get_similar_length_paths(self, single_paths, max_length: float = 5.):
        similar_length_paths = []
        for path in single_paths:
            if path.length() < self.individual_min_length:
                continue
            try:
                segments = self.crop_path_into_segments(path, length=max_length)
                similar_length_paths.extend(segments)
            except AssertionError:
                print("Error while cropping path into segments, skipping...")
                continue
        return similar_length_paths
    
    def get_similar_length_paths_from_index(self, index, max_length: float = 5.):
        svg_path = self.split[index]
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        sim_length_paths = self.get_similar_length_paths(single_paths, max_length=max_length)
        return sim_length_paths

    def __getitem__(self, index) -> tuple:
        svg_path = self.split[index]
        label = svg_path.split("/")[-2]
        label = string.printable.index(label)
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        # queue = copy.deepcopy(single_paths)
        sim_length_paths = self.get_similar_length_paths(single_paths, self.individual_max_length)
        assert check_for_continouity(sim_length_paths), "paths are not continous"
        # select a random slice of the paths of length max_shapes_per_svg
        if len(sim_length_paths) > self.max_shapes_per_svg:
            start_idx = random.randint(0, len(sim_length_paths) - self.max_shapes_per_svg)
            sim_length_paths = sim_length_paths[start_idx:start_idx+self.max_shapes_per_svg]
        rasterized_segments, centers = get_rasterized_segments(sim_length_paths, self.stroke_width, self.individual_max_length, svg_attributes, centered=True, height=self.width, width=self.width)
        imgs = torch.stack(rasterized_segments)  # (n_shapes, channels, width, width)
        centers = torch.tensor(centers)  # (n_shapes, 2)
        labels = torch.ones(imgs.size(0)) * label
        return imgs, labels.int(), centers
    
    def _get_full_item(self, index:int) -> List[Tensor]:
        """
        This function is intended to be used by the tokenization process.
        """
        svg_path = self.df.iloc[index]["file_path"]
        label = self.df.iloc[index]["class"]
        label = string.printable.index(label)
        description = self.df.iloc[index]["description"]

        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        # queue = copy.deepcopy(single_paths)
        sim_length_paths = self.get_similar_length_paths(single_paths, self.individual_max_length)
        assert check_for_continouity(sim_length_paths), "paths are not continous"
        rasterized_segments, centers = get_rasterized_segments(sim_length_paths, self.stroke_width, self.individual_max_length, svg_attributes, centered=True, height=self.width, width=self.width)
        imgs = torch.stack(rasterized_segments)  # (n_shapes, channels, width, width)
        centers = torch.tensor(centers)  # (n_shapes, 2)
        labels = torch.ones(imgs.size(0)) * label
        return imgs, labels.int(), centers, description
    
    def _get_full_svg_drawing(self, index, width:int = 720, as_tensor:bool = False):
        svg_path = self.split[index]
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        drawing = disvg(single_paths, paths2Drawing=True, stroke_widths=[self.stroke_width]*len(single_paths), viewbox = svg_attributes["viewBox"],dimensions=(width, width))
        if as_tensor:
            return svg_string_to_tensor(drawing.tostring())
        else:
            return drawing

    def __len__(self):
        return len(self.split)

class GlyphazznStage1Datamodule(LightningDataModule):
    def __init__(
        self,
        top_level_dir: str,
        train_batch_size: int,
        val_batch_size: int,
        channels: int,
        width: int,
        individual_max_length: float = 10.,
        max_shapes_per_svg:int=64,
        num_workers: int = 0,
        stroke_width: float = 0.3,
        subset:str = "all",
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.top_level_dir = top_level_dir
        self.channels = channels
        self.width = width
        self.num_workers = num_workers
        self.stroke_width = stroke_width
        self.individual_max_length = individual_max_length#
        self.subset = subset
        self.max_shapes_per_svg = max_shapes_per_svg

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = GlyphazznStage1Dataset(
            self.top_level_dir,
            self.channels,
            self.width,
            train=True,
            subset=self.subset,
            individual_max_length=self.individual_max_length,
            stroke_width=self.stroke_width,
            max_shapes_per_svg=self.max_shapes_per_svg,
        )

        self.val_dataset = GlyphazznStage1Dataset(
            self.top_level_dir,
            self.channels,
            self.width,
            train=False,
            subset=self.subset,
            individual_max_length=self.individual_max_length,
            stroke_width=self.stroke_width,
            max_shapes_per_svg=self.max_shapes_per_svg,
        )

    #       ===============================================================

    def collate_fn(self, batch):
        imgs, labels, centers = zip(*batch)
        imgs = torch.concat(imgs)
        labels = torch.concat(labels)
        centers = torch.concat(centers)
        return imgs, labels, centers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

class CenterShapeLayersFromSVGDataset(Dataset):
    """
    This dataset takes SVG files and preprocesses them into rasterized centered shape layers.

    Args:
        - csv_path: path to csv file with the following columns:
            - file_path: full path to svg file
            - class: class label
            - split: "train" or "test"
        - channels: number of channels for the rasterized images
        - width: width/height of the rasterized images
        - train: whether to use the train or test split
        - individual_min_length: minimum length of a path segment to qualify for being a single shape layer
        - individual_max_length: maximum length of a path segment, everything longer than this will be cropped into multiple segments
        - stroke_width: stroke width for rasterization
        - max_shapes_per_svg: maximum number of shape layers per svg file, can be tuned for VRAM savings
    """

    def __init__(self, 
                 csv_path: str,
                 channels: int,
                 width: int,
                 train: bool = True,
                 individual_min_length: float = 1.,
                 individual_max_length: float = 10.,
                 stroke_width: float = 0.3,
                 max_shapes_per_svg: int = 64,
                 **kwargs):
        super(CenterShapeLayersFromSVGDataset, self)
        self.csv_path = csv_path
        self.individual_min_length = individual_min_length
        self.individual_max_length = individual_max_length
        self.stroke_width = stroke_width
        self.max_shapes_per_svg = max_shapes_per_svg
        self.channels = channels
        self.width = width
        self.train = train
        self.split = pd.read_csv(self.csv_path)
        if train is not None:
            self.split = self.split[self.split["split"] == ("train" if self.train else "test")]
        else:
            print("[WARNING] Using the whole dataset! Train was None.")

    def crop_path_into_segments(self, path:Path, length:float = 5.):
        """
        a single input path is cropped into segments of approx length `length`. I say "approx" because we divide the path into same length segments, which will not be exactly `length` long.
        """
        segments = []
        num_iters = math.ceil(path.length() / length)
        for i in range(num_iters):
            cropped_segment = path.cropped(i/num_iters, (i+1)/num_iters)
            segments.append(cropped_segment)

        return segments

    def get_similar_length_paths(self, single_paths, max_length: float = 5.):
        similar_length_paths = []
        for path in single_paths:
            if path.length() < self.individual_min_length:
                continue
            try:
                segments = self.crop_path_into_segments(path, length=max_length)
                similar_length_paths.extend(segments)
            except AssertionError:
                print("Error while cropping path into segments, skipping...")
                continue
        return similar_length_paths
    
    def get_similar_length_paths_from_index(self, index, max_length: float = 5.):
        svg_path = self.split.iloc[index]["file_path"]
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        sim_length_paths = self.get_similar_length_paths(single_paths, max_length=max_length)
        return sim_length_paths

    def __getitem__(self, index) -> tuple:
        svg_path = self.split.iloc[index]["file_path"]
        label = self.split.iloc[index]["class"]
        label = string.printable.index(label)
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        # queue = copy.deepcopy(single_paths)
        sim_length_paths = self.get_similar_length_paths(single_paths, self.individual_max_length)
        assert check_for_continouity(sim_length_paths), "paths are not continous"
        # select a random slice of the paths of length max_shapes_per_svg
        if len(sim_length_paths) > self.max_shapes_per_svg:
            start_idx = random.randint(0, len(sim_length_paths) - self.max_shapes_per_svg)
            sim_length_paths = sim_length_paths[start_idx:start_idx+self.max_shapes_per_svg]
        rasterized_segments, centers = get_rasterized_segments(sim_length_paths, self.stroke_width, self.individual_max_length, svg_attributes, centered=True, height=self.width, width=self.width)
        imgs = torch.stack(rasterized_segments)  # (n_shapes, channels, width, width)
        centers = torch.tensor(centers)  # (n_shapes, 2)
        labels = torch.ones(imgs.size(0)) * label
        return imgs, labels, centers
    
    def _get_full_svg_drawing(self, index, width:int = 720):
        svg_path = self.split.iloc[index]["file_path"]
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        return disvg(single_paths, paths2Drawing=True, stroke_widths=[self.stroke_width]*len(single_paths), viewbox = svg_attributes["viewBox"],dimensions=(width, width))

    def __len__(self):
        return len(self.split)

class CenterShapeLayersFromSVGDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        train_batch_size: int,
        val_batch_size: int,
        channels: int,
        width: int,
        individual_max_length: float = 10.,
        num_workers: int = 0,
        stroke_width: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.csv_path = csv_path
        self.channels = channels
        self.width = width
        self.num_workers = num_workers
        self.stroke_width = stroke_width
        self.individual_max_length = individual_max_length

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = CenterShapeLayersFromSVGDataset(
            self.csv_path,
            self.channels,
            self.width,
            train=True,
            individual_max_length=self.individual_max_length,
            stroke_width=self.stroke_width
        )

        self.val_dataset = CenterShapeLayersFromSVGDataset(
            self.csv_path,
            self.channels,
            self.width,
            train=False,
            individual_max_length=self.individual_max_length,
            stroke_width=self.stroke_width
        )

    #       ===============================================================

    def collate_fn(self, batch):
        imgs, labels, centers = zip(*batch)
        imgs = torch.concat(imgs)
        labels = torch.concat(labels)
        centers = torch.concat(centers)
        return imgs, labels, centers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

class NewCausalSVGDataset(Dataset):
    """
    New FIGR8 dataset from a root directory for causal svg generation.
    Has:
        - centered images for prediction and visual MSE loss
        - absolute images for merged input
        - absolute and relative start/end points
        - stop signal vector


     ClassName
     |
     |
     |---I{svg_id}_{num_segments}_Segments_images_absolute.npy 
     |---I{svg_id}_{num_segments}_Segments_images_centered.npy 
     |---I{svg_id}_{num_segments}_Segments_positions.npy  
     |---split.csv
    """

    def __init__(self, root_path: str, context_length: int, channels: int, width: int, subset: List[str], **kwargs):
        super(NewCausalSVGDataset, self)
        self.context_length = context_length
        self.channels = channels
        self.width = width  # TODO: can we remove this?
        self.root_path = root_path
        self.subset = subset
        self.split = pd.read_csv(os.path.join(self.root_path, "split.csv"))
        before = len(self.split)
        self.split = self.split[self.split["segments"] < context_length]
        self.split = self.split[self.split["segments"] > 15]
        after = len(self.split)
        print(f"Removed {np.round((before - after) / before * 100, decimals=2)}% of samples because they have more segments than the context length.")
        if self.subset and len(self.subset) > 0:
            self.split = self.split[self.split['class'].isin(self.subset)]
        self.split = self.split[self.split["split"] == ("train" if kwargs["train"] else "test")]

    def __getitem__(self, index) -> tuple:
        centered_filename = self.split.iloc[index]["raster_filename_centered"]
        absolute_filename = self.split.iloc[index]["raster_filename_absolute"]
        positions_filename = self.split.iloc[index]["position_filename"]
        curr_class = self.split.iloc[index]["class"]

        centered_images = torch.from_numpy(np.load(os.path.join(self.root_path, curr_class, centered_filename)))
        absolute_images = torch.from_numpy(np.load(os.path.join(self.root_path, curr_class, absolute_filename)))
        positions = torch.from_numpy(np.load(os.path.join(self.root_path, curr_class, positions_filename))).to(torch.float32)
        # TODO test if this correct (expected: should extract start/end point of bezier in relative coordinates)
        positions = torch.flatten(positions[:, [2,3], :], start_dim=-2) # (t, 4, 2) -> (t, 4)

        num_timesteps = centered_images.shape[0]
        if centered_images.max() > 1:
            centered_images = centered_images / 255  # shift to [0-1] (imgs stored as uint8)
        if absolute_images.max() > 1:
            absolute_images = absolute_images / 255  # shift to [0-1] (imgs stored as uint8)

        # padding images with fewer features than CL
        pad_len = self.context_length - num_timesteps
        assert pad_len > 0, "context length must be greater than number of features of the dataset, did you set the context length correctly in the config?"
        pad = torch.ones(pad_len, *centered_images.shape[1:])  # 1 -> white -> no features
        
        centered_shape_layers = torch.concat((centered_images, pad), dim=0)  # Ground truth
        absolute_shape_layers = torch.concat((absolute_images, pad), dim=0)  # Ground truth
        
        position_pad = torch.ones(pad_len, *positions.shape[1:])
        positions = torch.concat((positions, position_pad), dim=0)

        # adding channel (Dataset is gray-scale, if you use 3 channels those are replicated)
        centered_shape_layers = centered_shape_layers[:, None].repeat((1, self.channels, 1, 1))
        absolute_shape_layers = absolute_shape_layers[:, None].repeat((1, self.channels, 1, 1))

        # merges absolute layers for full-image input
        merged_layers = absolute_shape_layers[0].unsqueeze(dim=0)
        for t in range(1 + 1, absolute_shape_layers.shape[0] + 1):
            merged_layers_t = torch.min(absolute_shape_layers[:t], dim = 0).values
            merged_layers = torch.cat((merged_layers, merged_layers_t.unsqueeze(dim=0)), dim=0)

        # input is all shifted one place to the right and starts with white canvas
        input_absolute_shape_layers = torch.concat((torch.ones(1, self.channels, *absolute_shape_layers.shape[-2:]), absolute_shape_layers[:-1]))
        input_centered_shape_layers = torch.concat((torch.ones(1, self.channels, *centered_shape_layers.shape[-2:]), centered_shape_layers[:-1]))
        input_merged_images = torch.concat((torch.ones(1, self.channels, *merged_layers.shape[-2:]), merged_layers[:-1]))
        
        # input is all shifted one place to the right and starts with pos (0, 0, 0, 0)
        input_positions = torch.concat((torch.zeros((1, 4)), positions[:-1]))

        # only take the positions of the first point as gt supervision signal
        gt_positions = positions[:,:2]

        # creating stop ground truth with 0: no stop, 1: stop, -1: padding
        stop_pad_len = pad_len - 1  # stop signals require one less padding than images
        stop_signals = torch.zeros(self.context_length)
        stop_signals[num_timesteps] = 1.
        if stop_pad_len >= 1:
            stop_signals[-stop_pad_len:] = -1.
        caption = f"black and white icon of a {self.split.iloc[index]['class']}"
        return input_absolute_shape_layers, input_centered_shape_layers, input_merged_images, stop_signals, caption, centered_shape_layers, input_positions, gt_positions

    def __len__(self):
        return len(self.split)

class NewCausalSVGDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int,
        val_batch_size: int,
        context_length: int,
        channels: int,
        width: int,
        num_workers: int = 0,
        subset: List = None,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.root_path = data_path
        self.context_length = context_length
        self.channels = channels
        self.width = width
        self.num_workers = num_workers
        self.subset = subset
        if subset:
            print(f"Using subset of original dataset: {self.subset}")
        else:
            print("Executing on the whole dataset!")

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = NewCausalSVGDataset(
            self.root_path,
            self.context_length,
            self.channels,
            self.width,
            subset=self.subset,
            train=True,
        )

        self.val_dataset = NewCausalSVGDataset(
            self.root_path,
            self.context_length,
            self.channels,
            self.width,
            subset=self.subset,
            train=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=64,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

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

        label = 0

        return image, label

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
        self._int_to_label = {}

        for i, label in enumerate(os.listdir(self.image_folder)):
            # important for CLIP sim later to have the real string label
            self._int_to_label[i] = label
            
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
        if image.mode == "RGBA":
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


class Figr8CausalSVGDataset(Dataset):
    """
    FIGR8 dataset from a root directory for causal svg generation

     Length
     |
     |---------------------------------------------|
     F{folder_id}_I{svg_id}_P{features}.npy    split.csv
    """

    def __init__(self, root_path: str, context_length: int, channels: int, width: int, subset: List, **kwargs):
        super(Figr8CausalSVGDataset, self)
        self.context_length = context_length
        self.channels = channels
        self.width = width  # TODO: can we remove this?
        self.root_path = root_path
        self.subset = subset
        self.split = pd.read_csv(os.path.join(self.root_path, "split.csv"))
        if self.subset and len(self.subset) > 0:
            self.split = self.split[self.split['class'].isin(self.subset)]
        self.split = self.split[self.split["split"] == ("train" if kwargs["train"] else "test")]

    def __getitem__(self, index) -> tuple:
        filename = self.split.iloc[index]["filename"]
        images = np.load(os.path.join(self.root_path, filename))
        images = torch.from_numpy(images)
        num_features = images.shape[0]
        if images.max() > 1:
            images = images / 255  # shift to [0-1] (imgs stored as uint8)

        # padding images with fewer features than CL
        pad_len = self.context_length - num_features
        assert pad_len > 0, "context length must be greater than number of features of the dataset, did you set the context length correctly in the config?"
        pad = torch.ones(pad_len, *images.shape[1:])  # 1 -> white -> no features
        shape_layers = torch.concat((images, pad), dim=0)  # Ground truth

        # adding channel (Dataset is gray-scale, if you use 3 channels those are replicated)
        shape_layers = shape_layers[:, None].repeat((1, self.channels, 1, 1))

        # merges layers for full-image supervision
        merged_layers = shape_layers[0].unsqueeze(dim=0)
        for t in range(1 + 1, shape_layers.shape[0] + 1):
            merged_layers_t = torch.min(shape_layers[:t], dim = 0).values
            merged_layers = torch.cat((merged_layers, merged_layers_t.unsqueeze(dim=0)), dim=0)

        # input is all shifted one place to the right and starts with white canvas
        images = torch.concat((torch.ones(1, self.channels, *images.shape[1:]), shape_layers[:-1]))
        merged_images = torch.concat((torch.ones(1, self.channels, *images.shape[-2:]), merged_layers[:-1]))

        # creating stop ground truth with 0: no stop, 1: stop, -1: padding
        stop_pad_len = pad_len - 1  # stop signals require one less padding than images
        stop_signals = torch.zeros(self.context_length)
        stop_signals[num_features] = 1.
        if stop_pad_len >= 1:
            stop_signals[-stop_pad_len:] = -1.
        caption = f"An image of {self.split.iloc[index]['class']}"
        return images, shape_layers, stop_signals, caption, merged_layers, merged_images

    def __len__(self):
        return len(self.split)


class DummyCausalSVGDataset(Dataset):
    """
    returns black dummy images as the full composite images that act as input 
    returns individual shape renderings for loss calculation. full image at index T requires the network to predict shape rendering at index T.
    returns stop signal vector
    """
    def __init__(self, context_length: int, channels: int, width: int, **kwargs):
        super(DummyCausalSVGDataset, self)
        self.context_length = context_length
        self.channels = channels
        self.width = width

    def __getitem__(self, index) -> Tensor:
        image = torch.ones((self.channels, self.width, self.width))
        stop_idx = torch.randint(1, (self.context_length - 1), (1,))

        stop_signals = torch.zeros(self.context_length)
        stop_signals[stop_idx:] = 1.

        images = torch.stack([image]*self.context_length, 0)
        shape_layers = images

        return images, shape_layers, stop_signals
    
    def __len__(self):
        return 500


class MNISTforCSVG(Dataset):
    """
    MNIST dataset from a root directory for causal svg generation.

    mnist
    |
    |--------------|
    training    testing
    |              |
    0-9           0-9
    """

    def __init__(self, root, context_length: int = 2, train=True, transform=None):
        super(MNISTforCSVG, self)
        assert context_length > 1, "context length must be greater than 1"
        self.root = root
        self.train = train
        self.transform = transform
        self.context_length = context_length

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
        
        width = image.size(1)
        white_image = torch.ones((3, width, width))
        input_images = torch.stack([white_image] + [image] + [white_image]*(max(self.context_length-2, 0)), dim=0)
        gt_shape_layers = torch.stack([image] + [white_image]*(self.context_length-1), dim=0)

        # 0 is not stop, 1 is stop, -1 is padding, makes masking easier
        stop_signals = torch.cat([torch.Tensor([0.]), torch.Tensor([1.]), torch.Tensor([-1.]*(max(self.context_length-2, 0)))], dim=0)

        return input_images, gt_shape_layers, stop_signals#, label

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


class CausalSVGDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int,
        val_batch_size: int,
        context_length: int,
        channels: int,
        width: int,
        num_workers: int = 0,
        subset: List = None,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.root_path = data_path
        self.context_length = context_length
        self.channels = channels
        self.width = width
        self.num_workers = num_workers
        self.subset = subset
        if subset:
            print(f"Using subset of original dataset: {self.subset}")
        else:
            print("Executing on the whole dataset!")

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = Figr8CausalSVGDataset(
            self.root_path,
            self.context_length,
            self.channels,
            self.width,
            subset=self.subset,
            train=True,
        )

        self.val_dataset = Figr8CausalSVGDataset(
            self.root_path,
            self.context_length,
            self.channels,
            self.width,
            subset=self.subset,
            train=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=100,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
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
        # TODO think about scaling [0.8, 1.2] and translating [-2.5, 2.5] from DeepSVG
        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(5.0, fill=256),
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


class DummyCausalSVGDataModule(LightningDataModule):
    def __init__(
        self,
        context_length: int, 
        channels: int, 
        width: int
    ):
        super().__init__()

        self.context_length = context_length
        self.channels = channels
        self.width = width

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = DummyCausalSVGDataset(
            self.context_length,
            self.channels,
            self.width,
            train=True,
        )

        self.val_dataset = DummyCausalSVGDataset(
            self.context_length,
            self.channels,
            self.width,
            train=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=8,
            num_workers=1,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=8,
            num_workers=1,
            shuffle=True,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=8,
            num_workers=1,
            shuffle=True,
            pin_memory=False,
        )


class MNISTDatasetCSVG(LightningDataModule):
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
        context_length: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.context_length = context_length

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

        self.train_dataset = MNISTforCSVG(
            self.data_dir,
            train=True,
            transform=train_transforms,
            context_length=self.context_length
        )

        self.val_dataset = MNISTforCSVG(
            self.data_dir,
            train=False,
            transform=val_transforms,
            context_length=self.context_length
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