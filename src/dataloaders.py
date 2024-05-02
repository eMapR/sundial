import datetime
import importlib
import lightning as L
import numpy as np
import os
import torch
import xarray as xr

from rioxarray import open_rasterio
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional

from pipeline.utils import clip_xy_xarray
from pipeline.settings import (
    load_yaml,
    CHIP_DATA_PATH,
    ANNO_DATA_PATH,
    STAT_DATA_PATH,
    TRAIN_SAMPLE_PATH,
    VALIDATE_SAMPLE_PATH,
    TEST_SAMPLE_PATH,
    PREDICT_SAMPLE_PATH,
)
from settings import DATALOADER_CONFIG


class ChipsDataset(Dataset):
    def __init__(self,
                 chip_size: int,
                 time_step: int | None,
                 file_type: str,
                 sample_path: str,
                 chip_data_path: str,
                 anno_data_path: str | None,
                 transform_config: Optional[dict] = {},
                 **kwargs):
        super().__init__()
        self.chip_size = chip_size
        self.time_step = time_step
        self.file_type = file_type
        self.sample_path = sample_path
        self.chip_data_path = chip_data_path
        self.anno_data_path = anno_data_path

        self.samples = np.load(self.sample_path)
        self.means = kwargs.get("means")
        self.stds = kwargs.get("stds")

        self._init_loaders(file_type)
        self._init_transformers(transform_config)
        self._num_transformers = len(
            self.transformers) if self.transformers else 1
        self._len = len(self.samples) * self._num_transformers

    def __getitem__(self, idx):
        # loading image idx
        sample_idx = idx // self._num_transformers
        transform_idx = idx % self._num_transformers
        if len(self.samples.shape) == 2:
            img_idx, time_idx = self.samples[sample_idx]
            slicer = slice(time_idx, self.time_step)
        else:
            img_idx = self.samples[sample_idx]
            slicer = slice(-self.time_step, None) if self.time_step else None

        # loading chip and slicing time if necessary
        chip = self.chip_loader(img_idx)
        if slicer is not None:
            chip = self.slice_time(chip, slicer)

        # converting to tensor
        chip = torch.as_tensor(chip.to_numpy(), dtype=torch.float)

        # reshaping gee data (D H W C) to pytorch format (C D H W)
        chip = chip.permute(3, 0, 1, 2)

        # transforming chip if specified
        if self.transformers:
            seed = datetime.datetime.now().timestamp()
            torch.manual_seed(seed)
            chip = self.transformers[transform_idx](chip)

        # preprocessing chip if specified
        if self.preprocessors:
            chip = self.preprocessors(chip)

        # including annotations if anno_data_path is set
        if self.anno_data_path is not None:
            annotation = self.get_annotation(sample_idx)
            if self.transformers and self.apply_to_anno[transform_idx]:
                torch.manual_seed(seed)
                annotation = self.transformers[transform_idx](annotation)
        else:
            annotation = torch.empty(0)
        return chip, annotation, img_idx

    def __len__(self):
        return self._len

    def _init_loaders(self, file_type: str):
        match file_type:
            case "zarr":
                self.chips = xr.open_zarr(self.chip_data_path)
                self.chip_loader = lambda name: self._zarr_loader(
                    self.chips, name)
                if self.anno_data_path is not None:
                    self.annos = xr.open_zarr(self.anno_data_path)
                    self.anno_loader = lambda name: self._zarr_loader(
                        self.annos, name)
            case "tif":
                self.chip_loader = lambda name: \
                    self._tif_loader(self.chip_data_path, name)
                if self.anno_data_path is not None:
                    self.anno_loader = lambda name: \
                        self._tif_loader(self.anno_data_path, name)

    def _zarr_loader(self, xarr: xr.Dataset, name: int):
        chip = xarr[str(name)]
        if self.chip_size < max(chip["x"].size, chip["y"].size):
            chip = clip_xy_xarray(chip, self.chip_size)
        return chip

    def _tif_loader(self, data_path: str, name: int):
        image_path = os.path.join(data_path, f"{name}.tif")
        with open_rasterio(image_path) as src:
            image = src.read()
        # TODO: implement multiple tif files for multiple time steps
        return image

    def _init_transformers(self, transform_config: dict):
        self.apply_to_anno = []
        self.transformers = []
        self.preprocessors = []

        if transform_config.get("include_original"):
            self.transformers.append(torch.nn.Identity())
            self.apply_to_anno.append(False)

        for transform in transform_config.get("transforms", []):
            class_path = transform.get("class_path")
            transform_path = class_path.rsplit(".", 1)
            match len(transform_path):
                case 1:
                    transform_cls = getattr(
                        importlib.import_module("transforms"), class_path)
                    transformer = transform_cls(
                        **transform.get("init_args", {}))
                case 2:
                    module_path, class_name = transform_path
                    transform_cls = getattr(
                        importlib.import_module(module_path), class_name)
                    transformer = transform_cls(
                        **transform.get("init_args", {}))
            if transform.get("preprocess"):
                self.preprocessors.append(transformer)
            else:
                self.transformers.append(transformer)
                self.apply_to_anno.append(
                    transform.get("apply_to_anno", False))

        composition = transform_config.get("composition", {})
        composition_path = composition.get("class_path")
        match composition_path:
            case None:
                self.preprocessors = torch.nn.Sequential(
                    *self.preprocessors)
            case _:
                module_path, class_name = composition_path.rsplit(".", 1)
                composition_cls = getattr(
                    importlib.import_module(module_path), class_name)
                self.transformers = [composition_cls(self.transformers)]
                self.preprocessors = [composition_cls(self.preprocessors)]
                self.apply_to_anno = [all(self.apply_to_anno)]
                self.num_transforms = 1

    def get_annotation(self, idx):
        annotation = self.anno_loader(str(self.samples[idx]))
        if self.chip_size < max(annotation["x"].size, annotation["y"].size):
            annotation = clip_xy_xarray(annotation, self.chip_size)
        return torch.as_tensor(annotation.to_numpy(), dtype=torch.float)

    def slice_time(self, xarr: xr.Dataset, slicer: slice):
        return xarr.sel(datetime=slicer)

    def indexed_transformer(self,
                            x: torch.tensor,
                            index: int):
        transform = self.transforms[index]
        return transform(x)


class ChipsDataModule(L.LightningDataModule):
    def __init__(
            self,
            batch_size: int = DATALOADER_CONFIG["batch_size"],
            num_workers: int = DATALOADER_CONFIG["num_workers"],
            chip_size: int = DATALOADER_CONFIG["chip_size"],
            time_step: int = DATALOADER_CONFIG["time_step"],
            file_type: str = DATALOADER_CONFIG["file_type"],
            transform_config: dict = DATALOADER_CONFIG["transform_config"],
            train_sample_path: str = TRAIN_SAMPLE_PATH,
            validate_sample_path: str = VALIDATE_SAMPLE_PATH,
            test_sample_path: str = TEST_SAMPLE_PATH,
            predict_sample_path: str = PREDICT_SAMPLE_PATH,
            chip_data_path: str = CHIP_DATA_PATH,
            anno_data_path: str = ANNO_DATA_PATH,
            stat_data_path: str | None = STAT_DATA_PATH):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chip_size = chip_size
        self.time_step = time_step
        self.file_type = file_type
        self.transform_config = transform_config
        self.train_sample_path = train_sample_path
        self.validate_sample_path = validate_sample_path
        self.test_sample_path = test_sample_path
        self.predict_sample_path = predict_sample_path
        self.chip_data_path = chip_data_path
        self.anno_data_path = anno_data_path
        self.stat_data_path = stat_data_path

        # loading means and stds from stat_data_path
        if stat_data_path and os.path.exists(self.stat_data_path):
            stats = load_yaml(self.stat_data_path)
            if not transform_config.get("transforms"):
                transform_config = {"transforms": []}
            means = stats.get("chip_means")
            stds = stats.get("chip_stds")
            transform_config["transforms"].insert(0, {
                "class_path": "GeoNormalization",
                "init_args": {
                    "means": means,
                    "stds": stds
                },
                "preprocess": True,
            })
        else:
            means = None
            stds = None

        self.dataset_config = {
            "chip_size": self.chip_size,
            "time_step": self.time_step,
            "file_type": self.file_type,
            "transform_config": self.transform_config,
            "chip_data_path": self.chip_data_path,
            "anno_data_path": self.anno_data_path,
        }

        self.dataloader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "drop_last": True,
            "prefetch_factor": 2,
            "persistent_workers": True
        }

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        match stage:
            case "fit":
                self.training_ds = ChipsDataset(
                    **self.dataset_config | {"sample_path": self.train_sample_path})

                validate_transform_config = {
                    "transforms": [t for t in self.transform_config.get("transforms", []) if t.get("preprocess")],
                }
                self.validate_ds = ChipsDataset(
                    **self.dataset_config | {
                        "sample_path": self.validate_sample_path,
                        "transform_config": validate_transform_config})

            case "validate":
                self.validate_ds = ChipsDataset(
                    **self.dataset_config | {"sample_path": self.validate_sample_path})

            case "test":
                self.test_ds = ChipsDataset(
                    **self.dataset_config | {"sample_path": self.test_sample_path})

            case "predict":
                self.predict_ds = ChipsDataset(
                    **self.dataset_config | {"sample_path": self.predict_sample_path, "anno_data_path": None})

    def train_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.training_ds, "shuffle": True})

    def val_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.validate_ds})

    def test_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.test_ds, "drop_last": False})

    def predict_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.predict_ds, "drop_last": False})
