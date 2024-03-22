import lightning as L
import numpy as np
import os
import torch
import xarray as xr

from rioxarray import open_rasterio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional

from pipeline.utils import clip_xy_xarray
from pipeline.settings import (
    CHIP_DATA_PATH,
    ANNO_DATA_PATH,
    TRAIN_SAMPLE_PATH,
    VALIDATE_SAMPLE_PATH,
    TEST_SAMPLE_PATH,
    PREDICT_SAMPLE_PATH,
    SAMPLER,
    FILE_EXT_MAP
)


class PreprocesNormalization(nn.Module):
    def __init__(self,
                 means,
                 stds):
        super().__init__()
        self.means = torch.tensor(
            means, dtype=torch.float).view(-1, 1, 1, 1)
        self.stds = torch.tensor(
            stds, dtype=torch.float).view(-1, 1, 1, 1)

    def forward(self, x):
        return (x - self.means) / self.stds


class ChipsDataset(Dataset):
    def __init__(self,
                 chip_size: int,
                 year_step: int | None,
                 file_type: str,
                 sample_path: str,
                 chip_data_path: str,
                 anno_data_path: str | None,
                 means: Optional[list[float]] = None,
                 stds: Optional[list[float]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.chip_size = chip_size
        self.year_step = year_step
        self.file_type = file_type
        self.sample_path = sample_path
        self.chip_data_path = chip_data_path
        self.anno_data_path = anno_data_path
        self.means = means
        self.stds = stds

        self.normalizer = PreprocesNormalization(
            means, stds) if means and stds else None
        self.samples = np.load(self.sample_path)

        if self.file_type == "zarr":
            self.chips = xr.open_zarr(self.chip_data_path)
            self.chip_loader = lambda name: self._zarr_loader(self.chips, name)
            if self.anno_data_path is not None:
                self.annos = xr.open_zarr(self.anno_data_path)
                self.anno_loader = lambda name: self._zarr_loader(
                    self.annos, name)
        if self.file_type == "tif":
            self.chip_loader = lambda name: \
                self._tif_loader(self.chip_data_path, name)
            if self.anno_data_path is not None:
                self.anno_loader = lambda name: \
                    self._tif_loader(self.anno_data_path, name)

    def get_strata(self, idx):
        strata = self.anno_loader(str(self.samples[idx]))
        if self.chip_size < max(strata["x"].size, strata["y"].size):
            strata = clip_xy_xarray(strata, self.chip_size)
        return torch.as_tensor(strata.to_numpy(), dtype=torch.float)

    def slice_year(self, xarr: xr.Dataset, year_idx: int):
        return xarr.sel(year=slice(year_idx, year_idx+self.year_step))

    def __getitem__(self, idx):
        # loading image into xarr file and slicing if necessary
        if len(self.samples.shape) == 2 and self.year_step is not None:
            img_idx, year_idx = self.samples[img_idx]
            chip = self.chip_loader(str(img_idx))
            chip = self.slice_year(chip, year_idx)
        else:
            chip = self.chip_loader(str(self.samples[idx]))

        # clipping chip if larger than chip_size
        if self.chip_size < max(chip["x"].size, chip["y"].size):
            chip = clip_xy_xarray(chip, self.chip_size)

        # converting to tensor
        chip = torch.as_tensor(chip.to_numpy(), dtype=torch.float)

        # reshaping gee data (D H W C) to pytorch format (C D H W)
        chip = chip.permute(3, 0, 1, 2)

        # normalizing chip to precalculated means and stds
        if self.normalizer is not None:
            chip = self.normalizer(chip)

        # including annotations if anno_data_path is set
        if self.anno_data_path is not None:
            strata = self.get_strata(idx)
            return chip, strata, idx
        else:
            return chip, idx

    def __len__(self):
        return len(self.samples)

    def _zarr_loader(self, xarr: xr.Dataset, name: int):
        return xarr[name]

    def _tif_loader(self, data_path: str, name: int):
        image_path = os.path.join(data_path, f"{name}.tif")
        image = open_rasterio(image_path)
        return image


class ChipsDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        chip_size: int = SAMPLER["pixel_edge_size"],
        year_step: int = SAMPLER["year_step"],
        file_type: str = FILE_EXT_MAP[SAMPLER["file_type"]],
        train_sample_path: str = TRAIN_SAMPLE_PATH,
        validate_sample_path: str = VALIDATE_SAMPLE_PATH,
        test_sample_path: str = TEST_SAMPLE_PATH,
        predict_sample_path: str = PREDICT_SAMPLE_PATH,
        chip_data_path: str = CHIP_DATA_PATH,
        anno_data_path: str = ANNO_DATA_PATH,
        means: Optional[list[float]] = None,
        stds: Optional[list[float]] = None,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chip_size = chip_size
        self.year_step = year_step
        self.file_type = file_type
        self.train_sample_path = train_sample_path
        self.validate_sample_path = validate_sample_path
        self.test_sample_path = test_sample_path
        self.predict_sample_path = predict_sample_path
        self.chip_data_path = chip_data_path
        self.anno_data_path = anno_data_path
        self.means = means
        self.stds = stds

        self.dataset_config = {
            "chip_size": self.chip_size,
            "year_step": self.year_step,
            "file_type": self.file_type,
            "chip_data_path": self.chip_data_path,
            "anno_data_path": self.anno_data_path,
            "means": self.means,
            "stds": self.stds,
        }

        self.dataloader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "drop_last": True,
        }

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        match stage:
            case "fit":
                self.training_ds = ChipsDataset(
                    **self.dataset_config | {"sample_path": self.train_sample_path})

                self.validate_ds = ChipsDataset(
                    **self.dataset_config | {"sample_path": self.validate_sample_path})

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
            **self.dataloader_config | {"dataset": self.training_ds})

    def val_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.validate_ds})

    def test_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.test_ds})

    def predict_dataloader(self):
        return DataLoader(
            **self.dataloader_config | {"dataset": self.predict_ds})
