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
from typing import Literal
from shapely.ops import unary_union
from sklearn.model_selection import train_test_split

from logger import get_logger
from settings import RANDOM_STATE
from utils import parse_meta_data, generate_name, SQUARE_COLUMNS


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
        edge_size: int,
        projection: str | None) -> ee.feature:
    point = feature.geometry()
    maxError = ee.Number(0.001)
    match projection:
        case "UTM":
            # Generate UTM zone
            latitude = ee.Number(point.coordinates().get(0))
            longitude = ee.Number(point.coordinates().get(1))
            zone = ee.String(latitude.add(
                ee.Number(180)).divide(6).round().format("%d"))
            prefix = ee.String(ee.Algorithms.If(longitude.lt(
                0), ee.String("EPSG:326"), ee.String("EPSG:327")))
            projection = prefix.cat(zone)
        case None:
            # Note: repojection can be done at download
            maxError = None
        case _:
            projection = ee.Projection(projection)

    square = point.buffer(
        edge_size // 2).bounds(maxError=maxError, proj=projection).coordinates()
    return feature.set(
        "square",
        square)


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
        scale: int,
        projection: str | None) -> ee.FeatureCollection:
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
        projection=projection,
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


def generate_gaussian_squares(
        centroid_coords: tuple[float, float],
        edge_size: float,
        std_dev: float,
        num_squares: int) -> list[tuple[float, float]]:
    centroid_x, centroid_y = centroid_coords
    x_coords = np.random.normal(centroid_x, std_dev, num_squares)
    y_coords = np.random.normal(centroid_y, std_dev, num_squares)
    square_centroids = np.column_stack((x_coords, y_coords))
    squares = [generate_single_square(
        square_centroid, edge_size) for _ in range(num_squares) for square_centroid in square_centroids]

    return squares


def generate_single_square(
        centroid_coords: tuple[float, float],
        edge_size: float) -> list[tuple[float, float]]:
    dist_from_centroid = edge_size / 2
    centroid_x, centroid_y = centroid_coords
    return [
        (centroid_x - dist_from_centroid,
         centroid_y + dist_from_centroid),
        (centroid_x + dist_from_centroid,
         centroid_y + dist_from_centroid),
        (centroid_x + dist_from_centroid,
         centroid_y - dist_from_centroid),
        (centroid_x - dist_from_centroid,
         centroid_y - dist_from_centroid),
        (centroid_x - dist_from_centroid,
         centroid_y + dist_from_centroid)
    ]


def generate_squares(
        method: Literal["convering_grid", "random", "stratified", "single"],
        geo_file_path: str,
        meta_data_path: str,
        edge_size: int | float,
        num_points: int | None,
        num_strata: int | None,
        start_date: datetime | None,
        end_date: datetime | None,
        strata_scale: int | None,
        strata_columns: list[str] | None,
        projection: str | None,
        fraction: float | None) -> None:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    LOGGER.info("Loading geo file into GeoDataFrame...")
    gdf = gpd.read_file(geo_file_path)

    if method != "dataframe":
        unary_polygon = unary_union(gdf["geometry"])
        gee_polygon = ee.Geometry.Polygon(
            list(unary_polygon.exterior.coords))

    LOGGER.info(f"Generating squares via {method}...")
    match method:
        case "convering_grid":
            squares = gee_polygon.coveringGrid(
                scale=edge_size, proj=projection)
        case "random":
            points = generate_random_points(
                gee_polygon, edge_size, num_points)
            squares = points.map(
                lambda f: draw_bounding_square(f, edge_size, projection))
        case "stratified":
            points = stratified_sampling(
                num_points,
                num_strata,
                start_date,
                end_date,
                gee_polygon,
                strata_scale,
                projection)
            squares = points.map(
                lambda f: draw_bounding_square(f, edge_size, projection))
        case "single":
            if num_points is not None:
                groupby = gdf.groupby(strata_columns)
                sample = groupby.sample(n=num_points)
            elif fraction is not None:
                groupby = gdf.groupby(strata_columns)
                sample = groupby.sample(frac=fraction)
            else:
                sample = gdf
            sample = sample.reset_index(drop=True)
            points = ee.FeatureCollection([ee.Geometry.Point(
                list(p.centroid.coords)[0], proj=projection)
                for p in sample.loc[:, "geometry"]])
            squares = points.map(
                lambda f: draw_bounding_square(f, edge_size, projection))

    LOGGER.info("Downloading squares from GEE...")
    gdf = download_features(squares)

    LOGGER.info("Processing squares for zarr...")
    df_out = gdf.loc[:, "square"]\
        .explode()\
        .apply(pd.Series)\
        .add_prefix("square_")\
        .map(tuple)
    df_out.loc[:, "point"] = gdf.loc[:, "geometry"].apply(
        lambda p: p.coords[0])
    df_out.loc[:, "point_name"] = df_out.loc[:, "point"].apply(generate_name)
    df_out.loc[:, "square_name"] = df_out.loc[:, SQUARE_COLUMNS].apply(
        lambda r: generate_name(r.tolist()), axis=1)
    if "constant" in gdf.columns:
        df_out.loc[:, "constant"] = gdf.loc[:, "constant"]
    if strata_columns is not None:
        for col in strata_columns:
            df_out.loc[:, col] = sample.loc[:, col]

    LOGGER.info("Converting to xarray and saving...")
    xarr = df_out.to_xarray()
    coord_columns = SQUARE_COLUMNS + ["point"]
    xarr[coord_columns] = xarr[coord_columns].astype(
        [("x", float), ("y", float)])
    xarr.to_zarr(
        store=meta_data_path,
        mode="a")


def generate_time_combinations(
        start_year: int,
        end_year: int,
        back_step: int,
        meta_data_path: str) -> None:
    meta_data = xr.open_zarr(meta_data_path)
    years = range(end_year, start_year + back_step, -1)
    df_years = pd.DataFrame(years, columns=["year"])

    df_list = []
    for idx in range(meta_data.sizes["index"]):
        _, point_name, _, square_name, _, _ = parse_meta_data(
            meta_data, idx)
        new_df = df_years.copy()
        new_df.loc[:, "point_name"] = point_name
        new_df.loc[:, "square_name"] = square_name
        df_list.append(new_df)
    df_time = pd.concat(df_list, axis=0)
    df_time = df_time.reset_index(drop=True)

    return df_time


def split_sample_data(
        df_out: pd.DataFrame,
        training_ratio: float,
        test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(
        df_out, test_size=test_ratio, random_state=RANDOM_STATE)
    df_train, df_validate = train_test_split(
        df_train, test_size=training_ratio, random_state=RANDOM_STATE)
    return df_train, df_validate, df_test


def save_sample_data(
        df_train: pd.DataFrame,
        df_validate: pd.DataFrame,
        df_test: pd.DataFrame,
        training_sample_path: str,
        validate_sample_path: str,
        test_sample_path: str) -> None:
    df_list = [df_train, df_validate, df_test]
    paths = [training_sample_path, validate_sample_path, test_sample_path]
    for df, path in zip(df_list, paths):
        yarr = df.to_xarray()
        yarr.to_zarr(
            store=path,
            mode="a")


def main(**kwargs):
    from settings import SAMPLER as config, SAMPLE_PATH
    os.makedirs(SAMPLE_PATH, exist_ok=True)
    if (config_path := kwargs["config_path"]) is not None:
        with open(config_path, "r") as f:
            config = config | yaml.safe_load(f)
    global LOGGER
    LOGGER = get_logger(config["log_path"], config["log_name"])

    try:
        if kwargs["generate_squares"]:
            LOGGER.info("Generating squares...")
            start = time.time()
            square_config = {
                "method": config["method"],
                "geo_file_path": config["geo_file_path"],
                "edge_size": config["edge_size"],
                "num_points": config["num_points"],
                "num_strata": config["num_strata"],
                "start_date": config["start_date"],
                "end_date": config["end_date"],
                "strata_scale": config["strata_scale"],
                "strata_columns": config["strata_columns"],
                "projection": config["projection"],
                "fraction": config["fraction"],
                "meta_data_path": config["meta_data_path"],
            }
            generate_squares(**square_config)
            end = time.time()
            LOGGER.info(
                f"Square generation completed in: {(end - start)/60:.2} minutes")

        if kwargs["generate_time_sample"]:
            LOGGER.info("Generating time sample...")
            start = time.time()
            time_sample_config = {
                "start_year": config["start_date"].year,
                "end_year": config["end_date"].year,
                "back_step": config["back_step"],
                "meta_data_path": config["meta_data_path"],
            }
            df_out = generate_time_combinations(**time_sample_config)
            end = time.time()
            LOGGER.info(
                f"Time sample generation completed in: {(end - start)/60:.2} minutes")
        else:
            df_out = xr.open_zarr(config["meta_data_path"]).to_dataframe()

        LOGGER.info(
            "Splitting sample data into training, validation, and test sets...")
        df_train, df_validate, df_test = split_sample_data(
            df_out, config["training_ratio"], config["test_ratio"])

        LOGGER.info("Saving sample data splits to paths...")
        save_sample_data(
            df_train, df_validate, df_test,
            config["training_sample_path"],
            config["validate_sample_path"],
            config["test_sample_path"])

    except Exception as e:
        LOGGER.critical(f"Failed to generate sample: {type(e)} {e}")
        raise e


def parse_args():
    parser = argparse.ArgumentParser(description='Sampler Arguments')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--generate_squares', type=bool, default=True)
    parser.add_argument('--generate_time_sample', type=bool, default=True)
    return vars(parser.parse_args())


if __name__ == '__main__':
    main(**parse_args())
