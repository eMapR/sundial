import geopandas as gpd
import ee
import pandas as pd
import numpy as np
import os
import shapely
import time
import utm
import xarray as xr

from datetime import datetime
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import Polygon
from typing import Literal, Optional, Tuple


from constants import APPEND_DIM, CLASS_LABEL, DATETIME_LABEL


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


def get_class_weights(data: xr.DataArray) -> tuple[list[float], list[float]]:
    # TODO: fix dataset attribute error in annotator
    dims=[APPEND_DIM, "y", "x"]
    if DATETIME_LABEL in data.dims:
        dims.append(DATETIME_LABEL)
    sums = data.sum(dim=dims)
    class_totals = np.array(sums.values)
    class_probs = class_totals / class_totals.sum()
    if class_probs.sum() > 0:
        weights = 1 / class_probs
    else:
        weights = np.repeat(1, len(class_probs))
    return {"totals": class_totals.tolist(), "weights": weights.tolist()}


def get_band_stats(data: xr.DataArray) -> tuple[list[float], list[float]]:
    means = data.mean(dim=[APPEND_DIM, DATETIME_LABEL, "y", "x"])
    stds = data.std(dim=[APPEND_DIM, DATETIME_LABEL, "y", "x"])
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


def covering_grid(geo_dataframe: gpd.GeoDataFrame,
                    meter_edge_size: int,
                    overlap: int = 0,
                    year_offset: int = 0):
    xmin, ymin, xmax, ymax = geo_dataframe.total_bounds
    grid_cells = []
    for x0 in np.arange(xmin, xmax+meter_edge_size,  meter_edge_size * 1-overlap):
        for y0 in np.arange(ymin, ymax+meter_edge_size, meter_edge_size * 1-overlap):
            x1 = x0 - meter_edge_size
            y1 = y0 + meter_edge_size
            new_cell = shapely.geometry.box(x0, y0, x1, y1)
            if new_cell.intersects(geo_dataframe.geometry).any():
                grid_cells.append(new_cell)
            else:
                pass
    gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=geo_dataframe.crs)
    gdf.loc[:, DATETIME_LABEL] = max(geo_dataframe[DATETIME_LABEL].unique()) + year_offset
    return gdf