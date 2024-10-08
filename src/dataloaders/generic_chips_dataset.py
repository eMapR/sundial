import geopandas as gpd
import importlib
import lightning as L
import numpy as np
import os
import re
import torch
import xarray as xr

from datetime import datetime
from rioxarray import open_rasterio
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional

from pipeline.utils import clip_xy_xarray
from pipeline.settings import (
    load_yaml,
    CHIP_DATA_PATH,
    ANNO_DATA_PATH,
    STAT_DATA_PATH,
    META_DATA_PATH,
    TRAIN_SAMPLE_PATH,
    VALIDATE_SAMPLE_PATH,
    TEST_SAMPLE_PATH,
    PREDICT_SAMPLE_PATH,
)
from settings import DATALOADER_CONFIG


class GenericChipsDataset(Dataset):
    def __init__(self,
                 chip_size: int,
                 time_step: int | None,
                 file_type: str,
                 sample_path: str,
                 chip_data_path: str,
                 anno_data_path: str | None,
                 split_tif: int | None = None,
                 extension_config: Optional[dict] = {},
                 transform_config: Optional[dict] = {},
                 preprocess_config: Optional[dict] = {},
                 **kwargs):
        super().__init__()
        self.chip_size = chip_size
        self.time_step = time_step
        self.file_type = file_type
        self.sample_path = sample_path
        self.chip_data_path = chip_data_path
        self.anno_data_path = anno_data_path
        self.split_tif = split_tif

        sample_type = os.path.splitext(self.sample_path)
        match sample_type[-1]:
            case ".npy":
                self.samples = np.load(self.sample_path)
            case ".txt" | ".text":
                with open(self.sample_path, 'r') as file:
                    self.samples = file.read().splitlines()
                
        self.means = kwargs.get("means")
        self.stds = kwargs.get("stds")

        self._init_loaders(self.file_type)
        self._init_extensions(extension_config)
        self._init_preprocessors(preprocess_config)
        self._init_transformers(transform_config)

    def __getitem__(self, idx):
        # loading image idx
        sample_idx = idx // self.num_transforms
        transform_idx = idx % self.num_transforms
        if not isinstance(self.samples, list) and len(self.samples.shape) == 2:
            img_name, time_idx = self.samples[sample_idx]
            slicer = slice(time_idx, time_idx + self.time_step)
        else:
            img_name = self.samples[sample_idx]
            slicer = slice(-self.time_step, None) if self.time_step else None
        
        # parsing img name for index
        if isinstance(img_name, str):
            img_idx = int(re.search(r'chip_(\d+)\.tif', img_name).group(1))
        else:
            img_idx = img_name

        # loading chip and slicing time if necessary
        chip = self.chip_loader(img_name)
        if slicer is not None:
            chip = chip[:, slicer, :, :]

        # preprocessing chip if specified
        if self.chip_preprocessor:
            chip = self.chip_preprocessor(chip)

        # transforming chip if specified
        if self.transformers:
            seed = datetime.now().timestamp()
            torch.manual_seed(seed)
            transform = self.transformers[transform_idx]["transform"]
            apply_to_anno = self.transformers[transform_idx]["apply_to_anno"]
            chip = transform(chip)

        # including annotations if anno_data_path is set
        if self.anno_data_path is not None:
            anno = self.anno_loader(img_name)
            # preprocessing chip if specified
            if self.anno_preprocessor:
                anno = self.anno_preprocessor(anno)
            if self.transformers and apply_to_anno:
                torch.manual_seed(seed)
                anno = transform(anno)
        else:
            anno = torch.empty(0)
        ret = (chip, anno, idx)
        if self.extensions:
            for ext in self.extensions:
                if ext.meta_data:
                    ret += (ext.get_item(img_idx, self.meta_data),)
                else:
                    ret += (ext.get_item(img_idx),)
        return ret

    def __len__(self):
        return len(self.samples) * self.num_transforms

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
                    self._tif_loader(self.chip_data_path, name, self.split_tif)
                if self.anno_data_path is not None:
                    self.anno_loader = lambda name: \
                        self._tif_loader(self.anno_data_path, name, None)

    def _zarr_loader(self, xarr: xr.Dataset, name: int):
        chip = xarr[str(name)]
        if self.chip_size < max(chip["x"].size, chip["y"].size):
            chip = clip_xy_xarray(chip, self.chip_size)
            
        return torch.tensor(chip.to_numpy(), dtype=torch.float)

    def _tif_loader(self, data_path: str, name: int, split_tif: int | None):
        image_path = os.path.join(data_path, name)
        with open_rasterio(image_path) as chip:
            if self.chip_size < max(chip["x"].size, chip["y"].size):
                chip = clip_xy_xarray(chip, self.chip_size)
            
            chip = torch.tensor(chip.to_numpy(), dtype=torch.float)
            if split_tif:
                chip = chip.reshape(-1, self.split_tif, chip.shape[-2], chip.shape[-1])
            
            return chip

    def _init_transformers(self, transform_config: dict):
        self.transformers = []

        if transform_config.get("include_original"):
            self.transformers.append({
                "transform": torch.nn.Identity(),
                "apply_to_anno": True
            })

        for transform in transform_config.get("transforms", []):
            transformer = self._dynamic_transform_import(transform)
            self.transformers.append({
                "transform": transformer,
                "apply_to_anno": transform.get("apply_to_anno", False)
            })

        composition = transform_config.get("composition", {})
        composition_path = composition.get("class_path")
        match composition_path:
            case None:
                pass
            case _:
                module_path, class_name = composition_path.rsplit(".", 1)
                composition_cls = getattr(
                    importlib.import_module(module_path), class_name)
                self.transformers = [{
                    "transform": composition_cls(*self.transformers),
                    "apply_to_anno": all([t["apply_to_anno"] for t in self.transformers])
                }]

        num_transforms = len(self.transformers)
        self.num_transforms = 1 if num_transforms == 0 else num_transforms

    def _init_preprocessors(self, preprocess_config: dict):
        chip = []
        anno = []

        for preprocess in preprocess_config.get("preprocesses", []):
            preprocessor = self._dynamic_transform_import(preprocess)
            targets = preprocess.get("targets", [])
            if "chip" in targets:
                chip.append(preprocessor)
            if "anno" in targets:
                anno.append(preprocessor)

        self.chip_preprocessor = torch.nn.Sequential(*chip) if chip else None
        self.anno_preprocessor = torch.nn.Sequential(*anno) if anno else None
        
    def _init_extensions(self, extension_config: dict):
        if extension_config.get("load_meta_data"):
            self.meta_data = gpd.read_file(META_DATA_PATH)
        extensions = extension_config.get("extensions")
        self.extensions = []
        for extension in extensions:
            self.extensions.append(self._dynamic_extension_import(extension))

    def _dynamic_transform_import(self, transform: dict):
        class_path = transform.get("class_path")
        transform_path = class_path.rsplit(".", 1)
        match len(transform_path):
            case 1:
                transform_cls = getattr(
                    importlib.import_module("transforms"), class_path)
                return transform_cls(
                    **transform.get("init_args", {}))
            case 2:
                module_path, class_name = transform_path
                transform_cls = getattr(
                    importlib.import_module(module_path), class_name)
                return transform_cls(
                    **transform.get("init_args", {}))
                
    def _dynamic_extension_import(self, loader: dict):
        class_path = loader.get("class_path")
        loader_path = class_path.rsplit(".", 1)
        module_path, class_name = loader_path
        loader_cls = getattr(
            importlib.import_module(module_path), class_name)
        return loader_cls(
            **loader.get("init_args", {}))
        

class LatLotFromMeta():
    meta_data = True
    
    def get_item(self, idx: int, meta_data: gpd.GeoDataFrame):
        point = meta_data.iloc[idx].geometry
        return torch.tensor([point.y, point.x], dtype=torch.float)


class YearDayFromMeta():
    meta_data = True
    
    def __init__(self,
                 year_col: str,
                 dates: list[str | datetime]):
        self.year_col = year_col
        self.dates = dates
    
    def get_day_of_year(self, month_day: str, year: int):
        date_str = f"{year}-{month_day}"
        date = datetime.strptime(date_str, "%Y-%m-%d")

        day_of_year = date.timetuple().tm_yday
        return day_of_year
    
    def get_item(self, idx: int, meta_data: gpd.GeoDataFrame):
        year = meta_data[self.year_col].iloc[idx]
        return torch.tensor([(year, self.get_day_of_year(date, year)) for date in self.dates], dtype=torch.float)


class GenericChipsDataModule(L.LightningDataModule):
    def __init__(
            self,
            batch_size: int = DATALOADER_CONFIG["batch_size"],
            num_workers: int = DATALOADER_CONFIG["num_workers"],
            chip_size: int = DATALOADER_CONFIG["chip_size"],
            time_step: int = DATALOADER_CONFIG["time_step"],
            file_type: str = DATALOADER_CONFIG["file_type"],
            split_tif: int | None = DATALOADER_CONFIG["split_tif"],
            extension_config: dict = DATALOADER_CONFIG["extension_config"],
            preprocess_config: dict = DATALOADER_CONFIG["preprocess_config"],
            transform_config: dict = DATALOADER_CONFIG["transform_config"],
            train_sample_path: str = TRAIN_SAMPLE_PATH,
            validate_sample_path: str = VALIDATE_SAMPLE_PATH,
            test_sample_path: str = TEST_SAMPLE_PATH,
            predict_sample_path: str = PREDICT_SAMPLE_PATH,
            chip_data_path: str = CHIP_DATA_PATH,
            anno_data_path: str = ANNO_DATA_PATH,
            stat_data_path: str | None = STAT_DATA_PATH,
            **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chip_size = chip_size
        self.time_step = time_step
        self.file_type = file_type
        self.split_tif = split_tif
        self.extension_config = extension_config
        self.preprocess_config = preprocess_config
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
            if not self.preprocess_config.get("preprocesses"):
                self.preprocess_config = {"preprocesses": []}
            means = stats.get("chip_means")
            stds = stats.get("chip_stds")
            self.preprocess_config["preprocesses"].insert(0, {
                "class_path": "GeoNormalization",
                "init_args": {
                    "means": means,
                    "stds": stds
                },
                "targets": ["chip"]
            })
        else:
            means = None
            stds = None

        self.dataset_config = {
            "chip_size": self.chip_size,
            "time_step": self.time_step,
            "file_type": self.file_type,
            "split_tif": self.split_tif,
            "extension_config": self.extension_config,
            "preprocess_config": self.preprocess_config,
            "chip_data_path": self.chip_data_path,
            "anno_data_path": self.anno_data_path,
        } | kwargs

        self.dataloader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "drop_last": False,
            "prefetch_factor": 2,
            "persistent_workers": True
        }

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        match stage:
            case "fit":
                self.training_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "sample_path": self.train_sample_path,
                        "transform_config": self.transform_config
                    })

                validate_preprocess_config = {
                    "preprocesses": [t for t in self.preprocess_config.get("preprocesses", []) if t.get("validate", True)],
                }

                self.validate_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "sample_path": self.validate_sample_path,
                        "preprocess_config": validate_preprocess_config})

            case "validate":
                self.validate_ds = GenericChipsDataset(
                    **self.dataset_config | {"sample_path": self.validate_sample_path})

            case "test":
                self.test_ds = GenericChipsDataset(
                    **self.dataset_config | {"sample_path": self.test_sample_path})

            case "predict":
                self.predict_ds = GenericChipsDataset(
                    **self.dataset_config | {"sample_path": self.predict_sample_path, "anno_data_path": None})

    def train_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.training_ds, "shuffle": True, "drop_last": True})

    def val_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.validate_ds})

    def test_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.test_ds})

    def predict_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.predict_ds})
