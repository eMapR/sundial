import geopandas as gpd
import ee
import pandas as pd
import numpy as np
import os
import time
import utm
import xarray as xr

from datetime import datetime
from typing import Literal, Optional

from pipeline.logger import get_logger
from pipeline.settings import (NO_DATA_VALUE,
                               LOG_PATH, METHOD,
                               DATETIME_LABEL,
                               RANDOM_SEED,
                               CLASS_LABEL)


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


def gee_get_ads_score_image(
        area_of_interest: ee.Geometry) -> ee.Image:
    return ee.Image(os.getenv("ADS_SCORE_IMAGE_LINK")).clip(area_of_interest)


def gee_get_elevation_image(
        area_of_interest: ee.Geometry) -> ee.Image:
    return ee.Image('USGS/SRTMGL1_003').clip(area_of_interest)


def gee_get_prism_image(
        area_of_interest: ee.Geometry,
        start_date: datetime,
        end_date: datetime) -> ee.Image:
    collection = ee.ImageCollection("OREGONSTATE/PRISM/AN81m")\
        .filterBounds(area_of_interest)\
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    return collection.reduce(ee.Reducer.mean()).select(["ppt_mean"], ["ppt"])


def gee_get_percentile_ranges(
        single_band_image: ee.Image,
        area_of_interest: ee.Geometry,
        percentiles: ee.List) -> list[int]:
    return sorted(single_band_image.reduceRegion(
        reducer=ee.Reducer.percentile(percentiles),
        geometry=area_of_interest,
        maxPixels=1e13
    ).values().getInfo())


@function_timer
def gee_stratify_by_percentile(
        single_band_image: ee.Image,
        percentiles: list[int],
        out_band_name: str = None) -> ee.Image:
    result = ee.Image(0)
    for idx in range(len(percentiles) - 1):
        mask = single_band_image.gte(percentiles[idx]).And(
            single_band_image.lt(percentiles[idx+1]))
        result = result.where(mask, ee.Image(idx+1))
    if out_band_name is not None:
        result = result.select(["constant"], [out_band_name])
    return result


@function_timer
def gee_generate_random_points(
        feature: ee.Feature,
        radius: int,
        num_points: int,
) -> ee.FeatureCollection:
    geometry = feature.geometry().buffer(distance=radius)
    return ee.FeatureCollection.randomPoints(
        region=geometry,
        points=num_points,
        seed=RANDOM_SEED,
    )


@function_timer
def gee_stratified_sampling(
        num_points: int,
        num_classes: int,
        scale: int,
        start_date: datetime,
        end_date: datetime,
        sources: Literal["prism", "elevation", "ads_score"],
        area_of_interest: ee.Geometry,
        projection: str) -> ee.FeatureCollection:
    # creating percentiles for stratification
    num_images = len(sources)
    percentiles = ee.List.sequence(0, 100, count=num_classes+1)

    # Getting data images for stratification
    raw_images = []
    for source in sources:
        match source:
            case "prism":
                raw_images.append(
                    gee_get_prism_image(area_of_interest, start_date, end_date))
            case "elevation":
                raw_images.append(
                    gee_get_elevation_image(area_of_interest))
            case "ads_score":
                raw_images.append(
                    gee_get_ads_score_image(area_of_interest))
            case _:
                raise ValueError(f"Invalid source: {source}")

    # stratify by percentile
    stratified_images = []
    for image in raw_images:
        percentile_ranges = gee_get_percentile_ranges(
            image, area_of_interest, percentiles)
        stratified_images.append(
            gee_stratify_by_percentile(image, percentile_ranges))

    # concatenate stratified images
    if num_images == 1:
        population = stratified_images[0]
    else:
        combined = ee.Image.cat(stratified_images)
        num_bands = num_images
        concatenate_expression = " + ".join(
            [f"(b({i})*(100**{i}))" for i in range(num_bands)])
        population = combined.expression(concatenate_expression).toInt()

    # get stratified random sample experession for compute features laters
    return population.stratifiedSample(
        num_points,
        region=area_of_interest,
        scale=scale,
        projection=projection,
        geometries=True)


@function_timer
def gee_download_features(
        features: ee.FeatureCollection) -> gpd.GeoDataFrame:
    return ee.data.computeFeatures({
        "expression": features,
        "fileFormat": "GEOPANDAS_GEODATAFRAME"})

    
@function_timer
def stratified_sample(
        geo_dataframe: gpd.GeoDataFrame,
        num_points: Optional[float | int] = None):
    if num_points is not None:
        groupby = geo_dataframe.groupby(CLASS_LABEL)
        match num_points:
            case num if isinstance(num, float):
                sample = groupby.sample(frac=num)
            case num if isinstance(num, int):
                sample = groupby.sample(n=num)
    else:
        sample = geo_dataframe
    sample = sample.reset_index().rename(columns={'index': 'geo_file_index'})
    return sample


@function_timer
def generate_centroid_squares(
        geo_dataframe: gpd.GeoDataFrame,
        meter_edge_size: int) -> gpd.GeoDataFrame:
    geo_dataframe = geo_dataframe.reset_index()
    geo_dataframe.loc[:, "geometry"] = geo_dataframe.loc[:, "geometry"]\
        .apply(lambda p: p.centroid.buffer(meter_edge_size // 2).envelope)
    return geo_dataframe