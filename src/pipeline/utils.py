import pandas as pd
import numpy as np
import time
import utm
import xarray as xr

from datetime import datetime
from typing import Generator, Tuple

from pipeline.logger import get_logger
from pipeline.settings import NO_DATA_VALUE, LOG_PATH, METHOD, DATETIME_LABEL


def clip_xy_xarray(xarr: xr.DataArray, 
                   pixel_edge_size: int) -> xr.DataArray:
    x_diff = xarr["x"].size - pixel_edge_size
    y_diff = xarr["y"].size - pixel_edge_size
    
    assert x_diff > 0 and y_diff > 0, "image must be larger than clip size"

    x_start = x_diff // 2 
    x_end = xarr["x"].size - (x_diff - x_start)

    y_start = y_diff // 2
    y_end = xarr["y"].size - (y_diff - y_start)

    return xarr.sel(x=slice(x_start, x_end), y=slice(y_start, y_end))


def pad_xy_xarray(
        xarr: xr.DataArray,
        pixel_edge_size: int) -> xr.DataArray:
    x_diff = pixel_edge_size - xarr["x"].size
    y_diff = pixel_edge_size - xarr["y"].size

    assert x_diff > 0 and y_diff > 0, "image must be smaller than pad size"

    x_start = x_diff // 2
    x_end = x_diff - x_start

    y_start = y_diff // 2
    y_end = y_diff - y_start

    xarr = xarr.pad(
        x=(x_start, x_end),
        y=(y_start, y_end),
        keep_attrs=True,
        mode="constant",
        constant_values=NO_DATA_VALUE)
    return xarr


def generate_coords_name(coords: tuple[float], index) -> str:
    if len(coords) > 2:
        coords = coords[:-1]
    return f"{index}-" + "_".join([f"x{x}y{y}" for x, y in coords])


def get_utm_zone(point_coords: list[tuple[float]]) -> int:
    revserse_point = reversed(point_coords[0])
    utm_zone = utm.from_latlon(*revserse_point)[-2:]
    epsg_prefix = "EPSG:326" if point_coords[1] > 0 else "EPSG:327"
    epsg_code = f"{epsg_prefix}{utm_zone[0]}"

    return epsg_code


def function_timer(func):
    logger = get_logger(LOG_PATH, METHOD)

    def timer(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if logger:
            logger.info(
                f"{func.__name__} took {(end - start)/60:.3f} minutes to complete.")
        else:
            print(
                f"{func.__name__} took {(end - start)/60:.3f} minutes to complete.")
        return result
    return timer


def train_validate_test_split(samples: np.array, 
                              ratios: list[int],
                              random_seed: float | int) -> np.array:
    assert len(ratios) == 2 or len(ratios) == 3, "Ratios must be a list or array of 2 ors 3 elements (val, test) or (train, val, test)"
    assert (np.isclose(sum(ratios), 1.0) and len(ratios) == 3) or (sum(ratios) < 1.0 and len(ratios) == 2), "Ratios must sum to 1 if train is included or is < 1 otherwise"

    if len(ratios) == 2:
        ratios = (1 - sum(ratios),) + tuple(ratios)

    n_total = len(samples)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(samples)

    train_end = int(ratios[0] * n_total)
    val_end = train_end + int(ratios[1] * n_total)

    train = samples[:train_end,...]
    val = samples[train_end:val_end,...]
    test = samples[val_end:,...]

    return train, val, test

@function_timer
def get_chip_stats(data: xr.Dataset) -> dict:
    sums = data.sum(dim=data.dims).to_array()
    min_idx = sums.argmin().values
    max_idx = sums.argmax().values

    stats = {
        "mean": float(sums.mean().values),
        "std": float(sums.std().values),
        "min": float(sums[min_idx].values),
        "max": float(sums[max_idx].values),
        "count": len(data.variables)
    }

    return stats


@function_timer
def get_xarr_stats(data: xr.Dataset) -> dict:
    sums = data.sum(dim=data.dims).to_array()
    min_idx = sums.argmin().values
    max_idx = sums.argmax().values

    stats = {
        "mean": float(sums.mean().values),
        "std": float(sums.std().values),
        "min": float(sums[min_idx].values),
        "max": float(sums[max_idx].values),
        "count": len(data.variables)
    }

    return stats

@function_timer
def get_class_weights(data: xr.Dataset) -> tuple[list[float], list[float]]:
    sums = data.sum(dim=["y", "x"])
    totals = sums.to_dataarray().sum(dim="variable")
    weights = totals.min() / totals
    return {"totals": totals.values.tolist(), "weights": weights.values.tolist()}, sums


@function_timer
def get_band_stats(data: xr.Dataset) -> tuple[list[float], list[float]]:
    data = data.to_dataarray()
    means = data.mean(dim=["variable", DATETIME_LABEL, "y", "x"])
    stds = data.std(dim=["variable", DATETIME_LABEL, "y", "x"])
    return {"band_means": means.values.tolist(), "band_stds": stds.values.tolist()}