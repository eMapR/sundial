import geopandas as gpd
import ee
import pandas as pd
import numpy as np
import os
import time
import utm
import xarray as xr
import yaml

from datetime import datetime
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import Polygon
from typing import Literal, Optional, Tuple


def save_yaml(config: dict, path: str | os.PathLike):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        return config if config else {}


def update_yaml(config: dict, path: str | os.PathLike) -> dict:
    if os.path.exists(path):
        old_config = load_yaml(path)
        config = recursive_merge(old_config, config)
    save_yaml(config, path)


def recursive_merge(dict1: dict, dict2: dict):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_merge(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


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
        pixel_edge_size: int,
        no_data_value: float | int) -> xr.DataArray:
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
        constant_values=no_data_value)
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


def get_class_weights(data: xr.Dataset) -> tuple[list[float], list[float]]:
    class_totals = data.attrs["class_sums"].sum(axis=0)
    class_probs = class_totals / class_totals.sum()
    weights = 1 / class_probs
    return {"totals": class_totals.values.tolist(), "weights": weights.values.tolist()}


def get_band_stats(data: xr.Dataset,
                   datetime_label: str) -> tuple[list[float], list[float]]:
    data = data.to_dataarray()
    means = data.mean(dim=["variable", datetime_label, "y", "x"])
    stds = data.std(dim=["variable", datetime_label, "y", "x"])
    return {"band_means": means.values.tolist(), "band_stds": stds.values.tolist()}


def stratified_sample(
        geo_dataframe: gpd.GeoDataFrame,
        class_label: str,
        num_points: Optional[float | int] = None):
    if num_points is not None:
        groupby = geo_dataframe.groupby(class_label)
        match num_points:
            case num if isinstance(num, float):
                sample = groupby.sample(frac=num)
            case num if isinstance(num, int):
                sample = groupby.sample(n=num)
    else:
        sample = geo_dataframe
    sample = sample.reset_index().rename(columns={'index': 'geo_index'})
    return sample


def generate_centroid_squares(
        geo_dataframe: gpd.GeoDataFrame,
        meter_edge_size: int) -> gpd.GeoDataFrame:
    geo_dataframe.loc[:, "geometry"] = geo_dataframe.loc[:, "geometry"]\
        .apply(lambda p: p.centroid.buffer(meter_edge_size // 2).envelope)
    return geo_dataframe


def rasterizer(polygons: gpd.GeoSeries,
               square: Polygon,
               pixel_edge_size: int,
               fill: int | float,
               default_value: int | float):

    transform = from_bounds(*square.bounds, pixel_edge_size, pixel_edge_size)
    raster = rasterize(
        shapes=polygons,
        out_shape=(pixel_edge_size, pixel_edge_size),
        fill=fill,
        transform=transform,
        default_value=default_value)

    return raster