import os
import torch

import lightning as L
import xarray as xr

from torch.utils.data import Dataset, DataLoader
from typing import Literal

from utils.settings import DATALOADER as config


def zarr_loader(data_path: str, name: int, **kwargs):
    image = xr.open_zarr(data_path)[name]
    return image


def tif_loader(data_path: str, name: int, **kwargs):
    image_path = os.path.join(data_path, f"{name}.tif")
    image = xr.open_rasterio(image_path)
    return image


class ChipsDataset(Dataset):
    def __init__(self,
                 file_type: str,
                 chip_data_path: str,
                 sample_path: str,
                 chip_size: int,
                 base_year: int | None,
                 back_step: int | None,
                 mask_name: str | None,
                 transform=None,
                 **kwargs):
        super().__init__()
        self.file_type = file_type
        self.chip_data_path = chip_data_path
        self.sample_path = sample_path
        self.chip_size = chip_size
        self.base_year = base_year
        self.back_step = back_step
        self.mask_name = mask_name
        self.transform = transform

        self.image_loader = self._zarr_loader if self.file_type == "zarr" else self._zarr_loader

    def clip_chip(self, xarr):
        x_diff = xarr["x"].size - self.chip_size
        y_diff = xarr["y"].size - self.chip_size

        x_start = x_diff // 2
        x_end = x_diff - x_start

        y_start = y_diff // 2
        y_end = y_diff - y_start
        return xarr.sel(x=slice(x_start, -x_end), y=slice(y_start, -y_end))

    def get_mask(self, xarr):
        return xarr["overlap"].to_numpy()

    def slice_year(self, xarr, year):
        end_year = int(year) - self.base_year
        start_year = end_year - self.back_step
        # TODO: ensure coordinates are retained in xarray selection
        return xarr.sel(year=slice(start_year, end_year+1))

    def __getitem__(self, idx):
        # opening dataarray
        paths = xr.open_zarr(self.sample_path)
        name = paths["square_name"].isel(index=idx).values.item()
        year = paths["year"].isel(index=idx).values.item()
        image = self.image_loader(self.chip_data_path, name)

        if self.base_year is not None and self.back_step is not None:
            chip = self.slice_year(image, year)
        else:
            chip = image
        if self.chip_size < max(chip["x"].size, chip["y"].size):
            chip = self.clip_chip(chip)

        # TODO: normalize band values
        if self.transform:
            chip = self.transform(chip)

        if self.mask_name is not None:
            mask = self.get_mask(image)
            return torch.as_tensor(chip.to_numpy(), dtype=torch.float), torch.as_tensor(mask, dtype=torch.float)
        else:
            return torch.as_tensor(chip.to_numpy(), dtype=torch.float)

    def __len__(self):
        paths = xr.open_zarr(self.sample_path)
        return len(paths["index"])

    def _zarr_loader(self, data_path: str, name: int, **kwargs):
        image = xr.open_zarr(data_path)[name]
        return image

    def _tif_loader(self, data_path: str, name: int, **kwargs):
        image_path = os.path.join(data_path, f"{name}.tif")
        image = xr.open_rasterio(image_path)
        # TODO: convert to tensor
        return image


class ChipsDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        mask_name: str,
        file_type: str = config["file_type"],
        chip_data_path: str = config["chip_data_path"],
        train_sample_path: str = config["train_sample_path"],
        validate_sample_path: str = config["validate_sample_path"],
        test_sample_path: str = config["test_sample_path"],
        predict_sample_path: str = config["predict_sample_path"],
        chip_size: int = config["chip_size"],
        base_year: int = config["base_year"],
        back_step: int = config["back_step"],
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mask_name = mask_name
        self.file_type = file_type.lower()
        self.chip_data_path = chip_data_path
        self.train_sample_path = train_sample_path
        self.validate_sample_path = validate_sample_path
        self.test_sample_path = test_sample_path
        self.predict_sample_path = predict_sample_path
        self.chip_size = chip_size
        self.base_year = base_year
        self.back_step = back_step
        self.transform = None

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        config = {
            "file_type": self.file_type,
            "chip_data_path": self.chip_data_path,
            "chip_size": self.chip_size,
            "base_year": self.base_year,
            "back_step": self.back_step,
            "mask_name": self.mask_name,
            "transform": self.transform
        }

        match stage:
            case "fit":
                self.training_ds = ChipsDataset(
                    sample_path=self.train_sample_path,
                    **config)

                self.validate_ds = ChipsDataset(
                    sample_path=self.validate_sample_path,
                    **config)

            case "validate":
                self.validate_ds = ChipsDataset(
                    sample_path=self.validate_sample_path,
                    **config)

            case "test":
                self.test_ds = ChipsDataset(
                    sample_path=self.test_sample_path,
                    **config)

            case "predict":
                self.predict_ds = ChipsDataset(
                    sample_path=self.predict_sample_path,
                    **config)

    def train_dataloader(self):
        return DataLoader(
            self.training_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validate_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
