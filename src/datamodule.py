import torch

import lightning as L
import xarray as xr

from torch.utils.data import Dataset, DataLoader
from typing import Literal

from settings import DATAMODULE as config


class ChipsDataset(Dataset):
    def __init__(self,
                 chips_path: str,
                 sample_names_path: str,
                 chip_size: int,
                 base_year: int,
                 back_step: int,
                 transform=None):
        super().__init__()
        self.chips_path = chips_path
        self.sample_names_path = sample_names_path
        self.chip_size = chip_size
        self.base_year = base_year
        self.back_step = back_step
        self.transform = transform

    def clip_chip(self, xarr):
        x_diff = xarr["x"].size - self.chip_size
        y_diff = xarr["y"].size - self.chip_size

        x_start = x_diff // 2
        x_end = x_diff - x_start

        y_start = y_diff // 2
        y_end = y_diff - y_start
        return xarr.sel(x=slice(x_start, -x_end), y=slice(y_start, -y_end))

    def slice_year(self, xarr, year):
        end_year = int(year) - self.base_year
        start_year = end_year - self.back_step
        return xarr.sel(year=slice(start_year, end_year+1))

    def __getitem__(self, idx):
        # opening dataarray
        paths = xr.open_zarr(self.sample_names_path)
        name = paths["square"].isel(index=idx).values.item()
        year = paths["year"].isel(index=idx).values.item()
        chip = xr.open_zarr(self.chips_path)[name]

        # slicing target year and pixels
        chip = self.slice_year(self, chip, year)
        if self.chip_size < chip["x"].size:
            chip = self.clip_chip(chip)

        # TODO: normalize band values
        if self.transform:
            chip = self.transform(chip)

        return torch.as_tensor(chip, dtype=torch.float)

    def __len__(self):
        return len(self.chips_path)


class ChipsDataModule(L.LightningDataModule):
    MEAN = [
        # 6 values for bands
    ]

    STD = [
        # 6 values for bands
    ]

    def __init__(
        self,
        chip_data_path: str = config["chip_data_path"],
        training_sample_path: str = config["training_sample_path"],
        validate_sample_path: str = config["validate_sample_path"],
        test_sample_path: str = config["test_sample_path"],
        predict_sample_path: str = config["predict_sample_path"],
        chip_size: int = config["chip_size"],
        base_year: int = config["base_year"],
        back_step: int = config["back_step"],
        batch_size: int = config["batch_size"],
        num_workers: int = config["num_workers"],
    ):
        super().__init__()
        self.chip_data_path = chip_data_path
        self.training_sample_path = training_sample_path
        self.validate_sample_path = validate_sample_path
        self.test_sample_path = test_sample_path
        self.predict_sample_path = predict_sample_path
        self.chip_size = chip_size
        self.base_year = base_year
        self.back_step = back_step
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = None  # look into mean/std normalization

    def setup(self, stage: Literal["train", "validate", "test", "predict"]) -> None:
        config = {
            "chip_data_path": self.chip_data_path,
            "chip_size": self.chip_size,
            "base_year": self.base_year,
            "back_step": self.back_step,
            "transform": self.transform
        }

        match stage:
            case "train":
                self.training_ds = ChipsDataset(
                    sample_names_path=self.training_sample_path,
                    **config)
                
            case "validate":
                self.validate_ds = ChipsDataset(
                    sample_names_path=self.validate_sample_path,
                    **config)

            case "test":
                self.test_ds = ChipsDataset(
                    sample_names_path=self.test_sample_path,
                    **config)

            case "predict":
                self.predict_ds = ChipsDataset(
                    sample_names_path=self.predict_sample_path,
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
            self.validate_ds,
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
