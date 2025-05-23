import geopandas as gpd
import lightning as L
import numpy as np
import os
import pandas as pd
import re
import torch
import xarray as xr

from datetime import datetime
from rioxarray import open_rasterio
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from typing import Literal, Optional, Tuple

from config_utils import load_yaml
from constants import (
    CHIP_DATA_PATH,
    ANNO_DATA_PATH,
    STAT_DATA_PATH,
    META_DATA_PATH,
    TRAIN_SAMPLE_PATH,
    VALIDATE_SAMPLE_PATH,
    TEST_SAMPLE_PATH,
    PREDICT_SAMPLE_PATH,
)
from constants import APPEND_DIM, DATETIME_LABEL, CLASS_LABEL, IDX_NAME_ZFILL
from pipeline.utils import clip_xy_xarray
from settings import DATALOADER_CONFIG
from utils import dynamic_import


class GenericChipsDataset(Dataset):
    def __init__(self,
                 chip_size: int,
                 file_type: str,
                 window: Tuple[int, int],
                 sample_path: str,
                 chip_data_path: str,
                 anno_data_path: str | None,
                 split_tif: int | None,
                 class_indices: list[int] | None,
                 extension_config: Optional[dict] = {},
                 dynamic_transform_config: Optional[dict] = {},
                 static_transform_config: Optional[dict] = {},
                 **kwargs):
        super().__init__()
        self.chip_size = chip_size
        self.file_type = file_type
        self.window = window
        self.sample_path = sample_path
        self.chip_data_path = chip_data_path
        self.anno_data_path = anno_data_path
        self.split_tif = split_tif
        self.class_indices = class_indices
        self.extension_config = extension_config
        self.dynamic_transform_config = dynamic_transform_config
        self.static_transform_config = static_transform_config

        sample_type = os.path.splitext(self.sample_path)
        match sample_type[-1]:
            case ".npy":
                self.samples = np.load(self.sample_path)
            case ".txt" | ".text":
                with open(self.sample_path, 'r') as file:
                    self.samples = file.read().splitlines()
                
        self.means = kwargs.get("means")
        self.stds = kwargs.get("stds")

        self._init_loaders()
        self._init_extensions()
        self._init_static_transforms()
        self._init_dynamic_transforms()

    def __getitem__(self, indx):
        # loading image indx
        data = {}
        sample_indx = indx // len(self.dynamic_transforms)
        transform_indx = indx % len(self.dynamic_transforms)
        
        if not isinstance(self.samples, list) and len(self.samples.shape) == 2 and self.samples.shape[1] == 2:
            img_indx, time_indx = self.samples[sample_indx]
            img_indx, time_indx = int(img_indx), int(time_indx)
            time_slicer = slice(time_indx-self.window[0], time_indx+self.window[1])
            anno_time_indx = None
        elif not isinstance(self.samples, list) and len(self.samples.shape) == 2 and self.samples.shape[1] == 3:
            img_indx, time_indx, anno_time_indx = self.samples[sample_indx]
            img_indx, time_indx, anno_time_indx = int(img_indx), int(time_indx), int(anno_time_indx)
            time_slicer = slice(time_indx-self.window[0], time_indx+self.window[1])
        else:
            img_indx = self.samples[sample_indx]
            time_indx, anno_time_indx = None, None
            time_slicer = slice(*self.window) if self.window else slice(None)

        if isinstance(img_indx, str):
            img_indx = int(re.search(r'.*(\d+).*', img_indx).group(1))

        # parsing img name for index
        data["indx"] = img_indx
        if time_indx is not None:
            data["time_indx"] = time_indx
        if anno_time_indx is not None:
            data["anno_time_indx"] = anno_time_indx

        # loading chip and slicing/unsqueezing time if necessary
        sel = {APPEND_DIM: img_indx}
        if DATETIME_LABEL in self._chips.dims and time_slicer is not None:
            sel[DATETIME_LABEL] = time_slicer
        
        chip = self.chip_loader(sel)
        if len(chip.shape) == 3:
            chip = chip.unsqueeze(1)
        if self.chip_static_transforms is not None:
            chip = self.chip_static_transforms(chip)
        
        # including annotations if anno_data_path is set
        if self.anno_data_path is not None and os.path.exists(self.anno_data_path):
            sel = {APPEND_DIM: img_indx}
            if DATETIME_LABEL in self._annos.dims and anno_time_indx is not None:
                sel[DATETIME_LABEL] = anno_time_indx
            if self.class_indices is None:
                sel[CLASS_LABEL] = np.arange(*self._annos[CLASS_LABEL].shape)
            else:
                sel[CLASS_LABEL] = self.class_indices
            
            anno = self.anno_loader(sel)
            if self.anno_static_transforms is not None:
                anno = self.anno_static_transforms(anno)
        else:
            anno = torch.empty(0)

        if self.dynamic_transforms.empty:
            data["chip"] = chip
            data["anno"] = anno
        else:
            dynamic_transform = self.dynamic_transforms.iloc[transform_indx]["transform"]
            image_only = self.dynamic_transforms.iloc[transform_indx]["image_only"]
            if not image_only and anno.numel() > 0:
                seed = int(datetime.now().timestamp())
                torch.manual_seed(seed)
                data["chip"] = dynamic_transform(chip)
                torch.manual_seed(seed)
                data["anno"] = dynamic_transform(anno)
            else:
                data["chip"] = dynamic_transform(chip)
                data["anno"] = anno
        
        for ext in self.extensions:
            if ext.meta_data:
                ext_val = ext.get_item(img_indx, time_indx, self.meta_data)
            else:
                ext_val = ext.get_item(img_indx, time_indx)
            data[ext.name] = torch.tensor(ext_val, dtype=torch.float)
        return data

    def __len__(self):
        return len(self.samples) * len(self.dynamic_transforms)

    def _init_loaders(self):
        match self.file_type:
            case "zarr":
                self._chips = xr.open_dataarray(self.chip_data_path, engine='zarr', cache=False)
                self.chip_loader = lambda sel: self._zarr_loader(self._chips, sel)
                if self.anno_data_path is not None and os.path.exists(self.anno_data_path):
                    self._annos = xr.open_dataarray(self.anno_data_path, engine='zarr', cache=False)
                    self.anno_loader = lambda sel: self._zarr_loader(self._annos, sel)
            case "tif":
                self.chip_loader = lambda name: self._tif_loader(self.chip_data_path, name, self.split_tif)
                if self.anno_data_path is not None and os.path.exists(self.anno_data_path):
                    self.anno_loader = lambda name: self._tif_loader(self.anno_data_path, name, None)

    def _zarr_loader(self, xarr: xr.DataArray, sel: dict):
        chip = xarr.sel(sel)
        if self.chip_size < max(chip["y"].size, chip["x"].size):
            chip = clip_xy_xarray(chip, self.chip_size)
        chip = torch.tensor(chip.compute().values, dtype=torch.float)
        return chip

    def _tif_loader(self, data_path: str, name: int, split_tif: int | None):
        image_path = os.path.join(data_path, f"{name}.tif")
        with open_rasterio(image_path) as chip:
            if self.chip_size < max(chip["y"].size, chip["x"].size):
                chip = clip_xy_xarray(chip, self.chip_size)
                
            chip = torch.tensor(chip.values, dtype=torch.float)
            if split_tif:
                chip = chip.reshape(-1, self.split_tif, chip.shape[-2], chip.shape[-1])

            return chip

    def _init_dynamic_transforms(self):
        transform_list = []
        if self.dynamic_transform_config.get("include_original", True):
            transform_list.append({"transform": torch.nn.Identity(), "image_only": False})

        for t in self.dynamic_transform_config.get("transforms", []):
            transform_list.append({"transform": dynamic_import(t).forward, "image_only": t.get("image_only", True)})

        self.dynamic_transforms = pd.DataFrame(transform_list)
        
    def _init_static_transforms(self):
        chip_transforms = []
        anno_transforms = []

        for transform in self.static_transform_config.get("transforms", []):
            transform_obj = dynamic_import(transform)
            targets = transform.get("targets", ["chip", "anno"])
            if "chip" in targets:
                chip_transforms.append(transform_obj)
            if "anno" in targets:
                anno_transforms.append(transform_obj)

        self.chip_static_transforms = v2.Compose(chip_transforms) if chip_transforms else None
        self.anno_static_transforms = v2.Compose(anno_transforms) if anno_transforms else None
        
    def _init_extensions(self):
        if self.extension_config.get("load_meta_data"):
            self.meta_data = gpd.read_file(META_DATA_PATH)
        extensions = self.extension_config.get("extensions")
        self.extensions = []
        for extension in extensions:
            self.extensions.append(dynamic_import(extension))


class GenericChipsDataModule(L.LightningDataModule):
    def __init__(
            self,
            batch_size: int = DATALOADER_CONFIG["batch_size"],
            num_workers: int = DATALOADER_CONFIG["num_workers"],
            chip_size: int = DATALOADER_CONFIG["chip_size"],
            window: Tuple[int, int] = DATALOADER_CONFIG["window"],
            file_type: str = DATALOADER_CONFIG["file_type"],
            split_tif: int | None = DATALOADER_CONFIG["split_tif"],
            class_indices: list[int] | None = DATALOADER_CONFIG["class_indices"],
            extension_config: dict = DATALOADER_CONFIG["extension_config"],
            dataloader_config: dict = DATALOADER_CONFIG["dataloader_config"],
            static_transform_config: dict = DATALOADER_CONFIG["static_transform_config"],
            dynamic_transform_config: dict = DATALOADER_CONFIG["dynamic_transform_config"],
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
        self.window = window
        self.file_type = file_type
        self.split_tif = split_tif
        self.class_indices = class_indices
        self.extension_config = extension_config
        self.dataloader_config = dataloader_config
        self.static_transform_config = static_transform_config
        self.dynamic_transform_config = dynamic_transform_config
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
            if not self.static_transform_config.get("transforms"):
                self.static_transform_config["transforms"] = []
            if not self.extension_config.get("extensions"):
                self.extension_config["extensions"] = []
            
            means = stats.get("chip_stats")["band_means"]
            stds = stats.get("chip_stats")["band_stds"]
            self.static_transform_config["transforms"].insert(0, {
                "class_path": "transforms.GeoNormalization",
                "init_args": {
                    "means": means,
                    "stds": stds
                },
                "methods": ["all"],
                "targets": ["chip"]
            })
        else:
            means = None
            stds = None

        self.dataset_config = {
            "chip_size": self.chip_size,
            "window": self.window,
            "file_type": self.file_type,
            "split_tif": self.split_tif,
            "extension_config": self.extension_config,
            "chip_data_path": self.chip_data_path,
            "anno_data_path": self.anno_data_path,
            "class_indices": self.class_indices,
            "means": means,
            "stds": stds,
        }

        self.dataloader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": True,
            "pin_memory": True,
            "drop_last": True,
        } | dataloader_config

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        transforms = self.static_transform_config.get("transforms", [])
        match stage:
            case "fit":
                train_static_transform_config = {"transforms": self._filter_configs(transforms, "methods", ["all", "train"])}
                self.training_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "sample_path": self.train_sample_path,
                        "static_transform_config": train_static_transform_config,
                        "dynamic_transform_config": self.dynamic_transform_config,
                    })

                validate_static_transform_config = {"transforms": self._filter_configs(transforms, "methods", ["all", "validate"])}
                self.validate_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "sample_path": self.validate_sample_path,
                        "static_transform_config": validate_static_transform_config,
                    })

            case "validate":
                validate_static_transform_config = {"transforms": self._filter_configs(transforms, "methods", ["all", "validate"])}
                self.validate_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "sample_path": self.validate_sample_path,
                        "static_transform_config": validate_static_transform_config
                    })

            case "test":
                test_static_transform_config = {"transforms": self._filter_configs(transforms, "methods", ["all", "test"])}
                self.test_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "sample_path": self.test_sample_path,
                        "static_transform_config": test_static_transform_config
                    })

            case "predict":
                predict_static_transform_config = {"transforms": self._filter_configs(transforms, "methods", ["all", "predict"])}
                self.predict_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "sample_path": self.predict_sample_path,
                        "static_transform_config": predict_static_transform_config})

    def train_dataloader(self):
        return DataLoader(**self.dataloader_config | {"dataset": self.training_ds, "shuffle": True, "drop_last": True})

    def val_dataloader(self):
        return DataLoader(**self.dataloader_config | {"dataset": self.validate_ds})

    def test_dataloader(self):
        return DataLoader(**self.dataloader_config | {"dataset": self.test_ds})

    def predict_dataloader(self):
        return DataLoader(**self.dataloader_config | {"dataset": self.predict_ds})
        
    def _filter_configs(self, configs, label, filters):
        return [c for c in configs if (set(filters) & set(c.get(label, ["all"])))]
                    
