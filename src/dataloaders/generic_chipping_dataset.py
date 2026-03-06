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
    IMAGERY_PATH,
    ANNOTATIONS_PATH,
    STAT_DATA_PATH,
)
from settings import DATAMODULE_CONFIG
from config_utils import dynamic_import


class GenericChippingDataset(Dataset):
    def __init__(self,
                 split: Literal["fit", "validate", "test", "predict"],
                 imagery_path: str,
                 annotations_path: str | None,
                 sampler: dict,
                 dynamic_transform_config: Optional[dict],
                 static_transform_config: Optional[dict],
                 **kwargs):
        super().__init__()
        self.split = split
        self.imagery_path = imagery_path
        self.annotations_path = annotations_path
        self.sampler = sampler
        
        self.dynamic_transform_config = dynamic_transform_config if dynamic_transform_config else {}
        self.static_transform_config = static_transform_config if static_transform_config else {}

        self.means = kwargs.get("means")
        self.stds = kwargs.get("stds")
        
        self._init_sampler()
        self._init_static_transforms()
        self._init_dynamic_transforms()


    def __getitem__(self, indx):
        data = {}

        chip, anno, meta = self._sampler(indx)
        data["meta"] = meta
        
        if self.chip_static_transforms is not None:
            chip = self.chip_static_transforms(chip)

        if anno.numel() and self.anno_static_transforms is not None:
            anno = self.anno_static_transforms(anno)

        if self.dynamic_transforms.empty:
            data["chip"] = chip
            data["anno"] = anno
        else:
            chc = v2.RandomChoice(self.dynamic_transforms)
            dynamic_transform = chc["transform"]
            image_only = chc["image_only"]
            if not image_only and anno.numel():
                seed = int(datetime.now().timestamp())
                torch.manual_seed(seed)
                data["chip"] = dynamic_transform(chip)
                torch.manual_seed(seed)
                data["anno"] = dynamic_transform(anno)
            else:
                data["chip"] = dynamic_transform(chip)
                data["anno"] = anno

        return data

    def __len__(self):
        return len(self._sampler)

    def _init_sampler(self):
        imagery = xr.open_dataarray(self.imagery_path, engine='zarr', cache=False)
        if self.annotations_path is not None and os.path.exists(self.annotations_path):
            annotations = xr.open_dataarray(self.annotations_path, engine='zarr', cache=False)
        else:
            annotations = None
        self._sampler = dynamic_import(self.sampler, {"split": self.split, "imagery_da": imagery, "annotations_da": annotations})

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


class GenericChipsDataModule(L.LightningDataModule):
    def __init__(
            self,
            batch_size: int = DATAMODULE_CONFIG["batch_size"],
            num_workers: int = DATAMODULE_CONFIG["num_workers"],
            dataloader_config: dict = DATAMODULE_CONFIG["dataloader_config"],
            static_transform_config: dict = DATAMODULE_CONFIG["static_transform_config"],
            dynamic_transform_config: dict = DATAMODULE_CONFIG["dynamic_transform_config"],
            imagery_path: str = IMAGERY_PATH,
            annotations_path: str = ANNOTATIONS_PATH,
            stat_data_path: str | None = STAT_DATA_PATH):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataloader_config = dataloader_config
        self.static_transform_config = static_transform_config
        self.dynamic_transform_config = dynamic_transform_config
        self.imagery_path = imagery_path
        self.annotations_path = annotations_path
        self.stat_data_path = stat_data_path

        # loading means and stds from stat_data_path
        if stat_data_path and os.path.exists(self.stat_data_path):
            stats = load_yaml(self.stat_data_path)
            if not self.static_transform_config.get("transforms"):
                self.static_transform_config["transforms"] = []
            
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
            "imagery_path": self.imagery_path,
            "annotations_path": self.annotations_path,
            "sampler": self.sampler,
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
                        "split": "train",
                        "static_transform_config": train_static_transform_config,
                        "dynamic_transform_config": self.dynamic_transform_config,
                    })

                validate_static_transform_config = {"transforms": self._filter_configs(transforms, "methods", ["all", "validate"])}
                self.validate_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "split": "validate",
                        "static_transform_config": validate_static_transform_config,
                    })

            case "validate":
                validate_static_transform_config = {"transforms": self._filter_configs(transforms, "methods", ["all", "validate"])}
                self.validate_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "static_transform_config": validate_static_transform_config
                    })

            case "test":
                test_static_transform_config = {"transforms": self._filter_configs(transforms, "methods", ["all", "test"])}
                self.test_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "split": "test"
                        "static_transform_config": test_static_transform_config
                    })

            case "predict":
                predict_static_transform_config = {"transforms": self._filter_configs(transforms, "methods", ["all", "predict"])}
                self.predict_ds = GenericChipsDataset(
                    **self.dataset_config | {
                        "split": "predict"
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
                    
