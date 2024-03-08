import ee
import argparse
import os
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Literal
from shapely.ops import unary_union

from .utils import parse_meta_data, generate_coords_name
from .logger import get_logger
from .settings import RANDOM_STATE, SQUARE_COLUMNS, STRATA_ATTR_NAME, GEE_REQUEST_LIMIT, GEE_FEATURE_LIMIT


def get_elevation_image(
        area_of_interest: ee.Geometry) -> ee.Image:
    return ee.Image('USGS/SRTMGL1_003').clip(area_of_interest)


def get_prism_image(
        area_of_interest: ee.Geometry,
        start_date: datetime,
        end_date: datetime) -> ee.Image:
    collection = ee.ImageCollection("OREGONSTATE/PRISM/AN81m")\
        .filterBounds(area_of_interest)\
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    return collection.reduce(ee.Reducer.mean()).select(["ppt_mean"], ["ppt"])


def get_percentile_ranges(
        single_band_image: ee.Image,
        area_of_interest: ee.Geometry,
        percentiles: list[int]) -> list[int]:
    """
    Calculates the percentile ranges between 0 and 100 of a single band image within a specified area of interest.

    Args:
        single_band_image (ee.Image): The single band image.
        area_of_interest (ee.Geometry): The area of interest.
        percentiles (list[int]): The list of percentiles to calculate.

    Returns:
        list[int]: The sorted list of percentile ranges.
    """
    return sorted(single_band_image.reduceRegion(
        reducer=ee.Reducer.percentile(percentiles),
        geometry=area_of_interest,
        maxPixels=1e13
    ).values().getInfo())


def stratify_by_percentile(
        single_band_image: ee.Image,
        percentiles: list[int],
        out_band_name: str = None) -> ee.Image:
    """
    Stratifies a single-band image into different classes based on list of percentile ranges.

    Args:
        single_band_image (ee.Image): The single-band image to be stratified.
        percentiles (list[int]): A list of percentiles to use for stratification.
        out_band_name (str, optional): The name of the output band. Defaults to None.

    Returns:
        ee.Image: The stratified image.

    """
    result = ee.Image(0)
    for idx in range(len(percentiles) - 1):
        mask = single_band_image.gte(percentiles[idx]).And(
            single_band_image.lt(percentiles[idx+1]))
        result = result.where(mask, ee.Image(idx+1))
    if out_band_name is not None:
        result = result.select(["constant"], [out_band_name])
    return result


def draw_bounding_square(
        feature: ee.Feature,
        meter_edge_size: int) -> ee.feature:
    maxError = ee.Number(0.01)
    geometry = feature.geometry()
    point = ee.Geometry(
        ee.Algorithms.If(
            geometry.type().equals("Point"),
            geometry,
            geometry.centroid(maxError=maxError)
        ))
    square_coords = point.buffer(
        meter_edge_size // 2).bounds(maxError=maxError).coordinates()
    point_coords = point.coordinates()
    return feature.set({
        "square_coords": square_coords,
        "point_coords": point_coords
    })


def generate_random_points(
        feature: ee.Feature,
        radius: int,
        num_points: int,
) -> ee.FeatureCollection:
    geometry = feature.geometry().buffer(distance=radius)
    return ee.FeatureCollection.randomPoints(
        region=geometry,
        points=num_points,
        seed=RANDOM_STATE,
    )


def stratified_sampling(
        num_points: int,
        num_strata: int,
        start_date: datetime,
        end_date: datetime,
        area_of_interest: ee.Geometry,
        scale: int) -> ee.FeatureCollection:
    """
    Performs stratified sampling on images within a specified area of interest based on PRISM and elevation.

    Args:
        num_points (int): The total number of points to sample.
        num_strata (int): The total number of strata to divide the data into.
        start_date (datetime): The start date of the time range for image collection.
        end_date (datetime): The end date of the time range for image collection.
        area_of_interest (ee.Geometry): The area of interest for sampling.
        scale (int): The scale at which to perform stratified sampling.

    Returns:
        ee.FeatureCollection: A feature collection containing the sampled points.
    """
    num_images = 2
    num_strata = num_strata ** (1/num_images)
    num_points = num_points // num_strata
    percentiles = ee.List.sequence(0, 100, count=num_strata+1)

    LOGGER.info("Getting mean images for stratified sampling...")
    raw_images = []
    raw_images.append(get_prism_image(area_of_interest, start_date, end_date))
    raw_images.append(get_elevation_image(area_of_interest))

    LOGGER.info("Stratifying images by percentile...")
    stratified_images = []
    for image in raw_images:
        percentile_ranges = get_percentile_ranges(
            image, area_of_interest, percentiles)
        stratified_images.append(
            stratify_by_percentile(image, percentile_ranges))

    LOGGER.info("Concatenating stratified image class bands...")
    combined = ee.Image.cat(stratified_images)
    num_bands = num_images
    concatenate_expression = " + ".join(
        [f"(b({i})*(100**{i}))" for i in range(num_bands)])
    concatenated = combined.expression(concatenate_expression).toInt()

    LOGGER.info("Sending request for stratified sampling...")
    return concatenated.stratifiedSample(
        num_points,
        region=area_of_interest,
        scale=scale,
        geometries=True)


def download_features(features: ee.FeatureCollection) -> gpd.GeoDataFrame:
    try:
        return ee.data.computeFeatures({
            "expression": features,
            "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    except Exception as e:
        LOGGER.critical(
            f"Failed to convert feature collection to GeoDataFrame: {e}")
        raise e


def process_download(
        gdf_download: gpd.GeoDataFrame,
        sample: gpd.GeoDataFrame,
        strata_columns: list[str]) -> gpd.GeoDataFrame:
    # extracting square coordinates
    df_out = gdf_download.loc[:, "square_coords"]\
        .explode()\
        .apply(pd.Series)\
        .add_prefix("square_")\
        .map(tuple)

    # extracting geometry coordinates
    match gdf_download.loc[0:0, "geometry"].item().geom_type:
        case "Polygon":
            def f(p): return list(p.exterior.coords)
        case "Point":
            def f(p): return list(p.coords)
    df_out = pd.concat([df_out, gdf_download.loc[:, "geometry"]
                        .apply(f)
                        .apply(pd.Series)
                        .add_prefix("geometry_")], axis=1)
    geometry_columns = df_out.filter(like="geometry_").columns.to_list()

    # setting names and poin coordinates
    df_out.loc[:, "point_coords"] = gdf_download\
        .loc[:, "point_coords"].apply(tuple)
    df_out.loc[:, "point_name"] = df_out\
        .loc[:, "point_coords"].apply(generate_coords_name)
    df_out.loc[:, "square_name"] = df_out\
        .loc[:, SQUARE_COLUMNS].apply(lambda r: generate_coords_name(r.tolist()), axis=1)

    # adding in attribute data
    if "constant" in gdf_download.columns:
        df_out.loc[:, "constant"] = gdf_download.loc[:, "constant"]
    if strata_columns is not None:
        if "year" in sample.columns:
            df_out.loc[:, "year"] = sample.loc[:, "year"]
        for col in strata_columns:
            df_out.loc[:, col] = sample.loc[:, col]
        df_out.loc[:, STRATA_ATTR_NAME] = sample.loc[:, strata_columns].astype(
            str).apply(lambda x: '__'.join(x).replace(" ", "_"), axis=1)
    return df_out, geometry_columns


def get_square_features(sample: gpd.GeoDataFrame, meter_edge_size: int) -> gpd.GeoDataFrame:
    points = ee.FeatureCollection([ee.Geometry.Polygon(
        list(p.exterior.coords))
        for p in sample.loc[:, "geometry"]])
    squares = points.map(
        lambda f: draw_bounding_square(f, meter_edge_size))
    gdf = download_features(squares)
    return gdf


def generate_squares(
        method: Literal["convering_grid", "random", "stratified", "single"],
        geo_file_path: str,
        meta_data_path: str,
        meter_edge_size: int | float,
        num_points: int | None,
        num_strata: int | None,
        start_date: datetime | None,
        end_date: datetime | None,
        strata_map_path: str | None,
        strata_scale: int | None,
        strata_columns: list[str] | None,
        fraction: float | None) -> None:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    LOGGER.info("Loading geo file into GeoDataFrame...")
    gdf = gpd.read_file(geo_file_path)

    if method != "single":
        unary_polygon = unary_union(gdf[strata_columns].to_crs(epsg=4326))
        gee_polygon = ee.Geometry.Polygon(
            list(unary_polygon.exterior.coords))

    LOGGER.info(f"Generating squares via {method}...")
    match method:
        case "convering_grid":
            squares = gee_polygon.coveringGrid(scale=meter_edge_size)
        case "random":
            points = generate_random_points(
                gee_polygon, meter_edge_size, num_points)
            squares = points.map(
                lambda f: draw_bounding_square(f, meter_edge_size, None))
            LOGGER.info("Downloading squares to dataframe from GEE...")
            gdf = download_features(squares)
        case "stratified":
            points = stratified_sampling(
                num_points,
                num_strata,
                start_date,
                end_date,
                gee_polygon,
                strata_scale)
            squares = points.map(
                lambda f: draw_bounding_square(f, meter_edge_size, None))
            LOGGER.info("Downloading squares to dataframe from GEE...")
            gdf = download_features(squares)
        case "single":
            if fraction is not None:
                groupby = gdf.groupby(strata_columns)
                sample = groupby.sample(frac=fraction)
            elif num_points is not None:
                groupby = gdf.groupby(strata_columns)
                sample = groupby.sample(n=num_points)
            else:
                sample = gdf
            sample = sample.reset_index(drop=True)
            sample.loc[:, "geometry"] = sample.loc[:, "geometry"].to_crs(
                epsg=4326)

            if (num_samples := len(sample)) > GEE_FEATURE_LIMIT:
                # partition sample into batches. numpy array split has a deprecated warning
                # this is done to maintain same order of samples for post processing
                rem = num_samples % GEE_REQUEST_LIMIT
                s = num_samples // GEE_REQUEST_LIMIT
                b = []
                for i in range(40):
                    if rem > 0:
                        rem -= 1
                        b.append(sample.iloc[s*i:s*(i+1) + 1])
                    else:
                        b.append(sample.iloc[s*i:s*(i+1)])

                # execute download in parallel
                LOGGER.info(
                    f"Downloading {num_samples} squares to dataframe from GEE using pool with batchsize {num_samples // GEE_REQUEST_LIMIT}...")
                with ThreadPool(processes=GEE_REQUEST_LIMIT) as p:
                    def f(b): return get_square_features(b, meter_edge_size)
                    result = p.map(f, b)
                    gdf = pd.concat(result, axis=0).reset_index(drop=True)
            else:
                points = ee.FeatureCollection([ee.Geometry.Polygon(
                    list(p.exterior.coords))
                    for p in sample.loc[:, "geometry"]])
                squares = points.map(
                    lambda f: draw_bounding_square(f, meter_edge_size))
                LOGGER.info("Downloading squares to dataframe from GEE...")
                gdf = download_features(squares)

            # final processing check to ensure gee didn't fail
            assert len(gdf) == len(sample)

    LOGGER.info("Processing dataframe columns for zarr...")
    df_out, geometry_columns = process_download(gdf, sample, strata_columns)
    if strata_columns is not None and strata_map_path is not None:
        strata_map = {v: k for k, v in enumerate(
            df_out[STRATA_ATTR_NAME].unique())}
        with open(strata_map_path, "w") as f:
            yaml.dump(strata_map, f, default_flow_style=False)

    LOGGER.info("Converting to xarray and saving...")
    xarr = df_out.to_xarray().drop_vars("index")
    coord_columns = SQUARE_COLUMNS + geometry_columns + ["point_coords"]
    xarr[coord_columns] = xarr[coord_columns].astype(
        [("x", float), ("y", float)])
    xarr.to_zarr(store=meta_data_path, mode="a")


def generate_time_combinations(
        start_year: int,
        end_year: int,
        back_step: int,
        meta_data_path: str) -> xr.Dataset:
    meta_data = xr.open_zarr(meta_data_path)
    years = range(end_year, start_year + back_step, -1)
    df_years = pd.DataFrame(years, columns=["year"])

    df_list = []
    for idx in range(meta_data.sizes["index"]):
        _, _, point_name, _, square_name, _, _, _ = parse_meta_data(
            meta_data, idx, back_step)
        new_df = df_years.copy()
        new_df.loc[:, "point_name"] = point_name
        new_df.loc[:, "square_name"] = square_name
        df_list.append(new_df)
    df_time = pd.concat(df_list, axis=0)
    df_time = df_time.reset_index(drop=True)

    return df_time.to_xarray().drop_vars("index")


def train_test_split_xarr(
        indices: np.array,
        test_ratio: float) -> tuple[xr.Dataset, xr.Dataset]:
    indices = np.arange(indices.size)
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1-test_ratio))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    train = xarr.isel(index=train_indices).drop_encoding()
    test = xarr.isel(index=test_indices).drop_encoding()

    return train, test


def main(**kwargs):
    from .settings import SAMPLER as config, SAMPLE_PATH
    os.makedirs(SAMPLE_PATH, exist_ok=True)

    global LOGGER
    LOGGER = get_logger(config["log_path"], config["log_name"])

    start_main = time.time()
    try:
        if config["generate_squares"]:
            LOGGER.info("Generating squares...")
            start = time.time()
            square_config = {
                "method": config["method"],
                "geo_file_path": config["geo_file_path"],
                "meta_data_path": config["meta_data_path"],
                "meter_edge_size": config["meter_edge_size"],
                "num_points": config["num_points"],
                "num_strata": config["num_strata"],
                "start_date": config["start_date"],
                "end_date": config["end_date"],
                "strata_map_path": config["strata_map_path"],
                "strata_scale": config["strata_scale"],
                "strata_columns": config["strata_columns"],
                "fraction": config["fraction"]
            }
            generate_squares(**square_config)
            end = time.time()
            LOGGER.info(
                f"Square generation completed in: {(end - start)/60:.2} minutes")

        if config["generate_time_combinations"]:
            LOGGER.info("Generating time sample...")
            start = time.time()
            time_sample_config = {
                "start_year": config["start_date"].year,
                "end_year": config["end_date"].year,
                "back_step": config["back_step"],
                "meta_data_path": config["meta_data_path"],
            }
            xarr_out = generate_time_combinations(**time_sample_config)
            end = time.time()
            LOGGER.info(
                f"Time sample generation completed in: {(end - start)/60:.2} minutes")
        else:
            xarr_out = xr.open_zarr(config["meta_data_path"])

        if config["generate_train_test_split"]:
            LOGGER.info(
                "Splitting sample data into training, validation, test, and predict sets...")
            xarr_train, xarr_validate = train_test_split_xarr(
                xarr_out, config["validate_ratio"])
            xarr_validate, xarr_test = train_test_split_xarr(
                xarr_validate, config["test_ratio"])
            xarr_test, xarr_predict = train_test_split_xarr(
                xarr_test, config["predict_ratio"])

            LOGGER.info("Saving sample data splits to paths...")
            xarr_list = [xarr_train, xarr_validate, xarr_test, xarr_predict]
            paths = [config["train_sample_path"],
                     config["validate_sample_path"],
                     config["test_sample_path"],
                     config["predict_sample_path"]]
            for xarr, path in zip(xarr_list, paths):
                xarr.to_zarr(store=path, mode="a")
        else:
            xarr_out.to_zarr(store=config["train_sample_path"], mode="a")

    except Exception as e:
        LOGGER.critical(f"Failed to generate sample: {type(e)} {e}")
        raise e
    end = time.time()
    LOGGER.info(
        f"Sample completed in: {(end - start_main)/60:.2} minutes")


if __name__ == '__main__':
    main()
