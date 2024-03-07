import lightning as L
import os
import torch
import xarray as xr

from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Literal

from pipeline.utils import clip_xy_xarray


class PreprocesNormalization(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        self.means = torch.tensor(
            means, dtype=torch.float).view(1, 1, 1, -1)
        self.stds = torch.tensor(stds, dtype=torch.float).view(1, 1, 1, -1)

    def forward(self, x):
        return (x - self.means) / self.stds


class ChipsDataset(Dataset):
    def __init__(self,
                 means: list[float] | None,
                 stds: list[float] | None,

                 chip_size: int,
                 base_year: int | None,
                 back_step: int | None,
                 file_type: str,

                 chip_data_path: str,
                 anno_data_path: str,
                 sample_path: str,

                 drop_duplicates: bool = list[str] | None,
                 **kwargs):
        super().__init__(**kwargs)
        self.chip_size = chip_size
        self.base_year = base_year
        self.back_step = back_step
        self.file_type = file_type

        self.chip_data_path = chip_data_path
        self.anno_data_path = anno_data_path
        self.sample_path = sample_path
        self.drop_duplicates = drop_duplicates

        self.normalize = PreprocesNormalization(
            means, stds) if means and stds else None
        self.meta_data = xr.open_zarr(self.sample_path).to_dataframe()
        self.meta_data = self.meta_data[["square_name", "year"]]

        if self.file_type == "zarr":
            self.chips = xr.open_zarr(self.chip_data_path)
            self.annos = xr.open_zarr(self.anno_data_path)
            self.chip_loader = lambda name: self._zarr_loader(self.chips, name)
            self.anno_loader = lambda name: self._zarr_loader(self.annos, name)
        if self.file_type == "tif":
            self.chip_loader = lambda name: \
                self._tif_loader(self.chip_data_path, name)
            self.anno_loader = lambda name: \
                self._tif_loader(self.anno_data_path, name)

        if drop_duplicates is not None:
            self.meta_data = self.meta_data\
                .drop_duplicates(subset=drop_duplicates, keep=False)

    def get_strata(self, name):
        strata = self.anno_loader(name)
        if self.chip_size < max(strata["x"].size, strata["y"].size):
            strata = clip_xy_xarray(strata, self.chip_size)
        return torch.as_tensor(strata.to_numpy(), dtype=torch.float)

    def slice_year(self, xarr: xr.Dataset, year: int):
        end_year = int(year) - self.base_year
        start_year = end_year - self.back_step
        return xarr.sel(year=slice(start_year, end_year+1))

    def __getitem__(self, idx):
        # loading image into xarr file
        name = self.meta_data.iloc[idx].loc["square_name"]
        year = self.meta_data.iloc[idx].loc["year"]
        chip = self.chip_loader(name)

        # slicing to target year if chip is larger and back_step is set
        if self.base_year is not None and self.back_step is not None:
            chip = self.slice_year(chip, year)

        # clipping chip if larger than chip_size
        if self.chip_size < max(chip["x"].size, chip["y"].size):
            chip = clip_xy_xarray(chip, self.chip_size)

        # converting to tensor
        chip = torch.as_tensor(chip.to_numpy(), dtype=torch.float)

        # normalizing chip to precalculated means and stds
        if self.normalize is not None:
            chip = self.normalize(chip)

        # including annotations if anno_data_path is set
        if self.anno_data_path is not None:
            strata = self.get_strata(name)
            return chip, strata
        else:
            return chip

    def __len__(self):
        return len(self.meta_data)

    def _zarr_loader(self, xarr: xr.Dataset, name: int):
        return xarr[name]

    def _tif_loader(self, data_path: str, name: int):
        image_path = os.path.join(data_path, f"{name}.tif")
        image = xr.open_rasterio(image_path)
        # TODO: convert to tensor
        return image


class ChipsDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        means: list[float] | None,
        stds: list[float] | None,

        chip_size: int | None,
        base_year: int | None,
        back_step: int | None,

        file_type: str,
        chip_data_path: str,
        anno_data_path: str,
        train_sample_path: str,
        validate_sample_path: str,
        test_sample_path: str,
        predict_sample_path: str,

        drop_duplicates: list[str] | None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.means = means
        self.stds = stds

        self.chip_size = chip_size
        self.base_year = base_year
        self.back_step = back_step

        self.file_type = file_type.lower()
        self.chip_data_path = chip_data_path
        self.anno_data_path = anno_data_path
        self.train_sample_path = train_sample_path
        self.validate_sample_path = validate_sample_path
        self.test_sample_path = test_sample_path
        self.predict_sample_path = predict_sample_path

        self.drop_duplicates = drop_duplicates

        self.dataset_config = {
            "means": self.means,
            "stds": self.stds,
            "chip_size": self.chip_size,
            "base_year": self.base_year,
            "back_step": self.back_step,
            "file_type": self.file_type,
            "chip_data_path": self.chip_data_path,
            "anno_data_path": self.anno_data_path,
            "drop_duplicates": self.drop_duplicates,
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
                    sample_path=self.train_sample_path,
                    **self.dataset_config)

                self.validate_ds = ChipsDataset(
                    sample_path=self.validate_sample_path,
                    **self.dataset_config)

            case "validate":
                self.validate_ds = ChipsDataset(
                    sample_path=self.validate_sample_path,
                    **self.dataset_config)

            case "test":
                self.test_ds = ChipsDataset(
                    sample_path=self.test_sample_path,
                    **self.dataset_config)

            case "predict":
                self.predict_ds = ChipsDataset(
                    sample_path=self.predict_sample_path,
                    **self.dataset_config | {
                        "anno_data_path": None,
                        "include_names": True,
                    })

    def train_dataloader(self):
        return DataLoader(
            dataset=self.training_ds,
            **self.dataloader_config
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validate_ds,
            **self.dataloader_config
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            **self.dataloader_config
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_ds,
            **self.dataloader_config | {"drop_last": False}
        )
