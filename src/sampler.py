import ee
import argparse
import time
import utm
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
from utils import parse_meta_data, SQUARE_COLUMNS


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
        projection: str | None = None) -> ee.feature:
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
        seed=round(time.time()),
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

    LOGGER.info("Stratifying images...")
    stratified_images = []
    for image in raw_images:
        percentile_ranges = get_percentile_ranges(
            image, area_of_interest, percentiles)
        stratified_images.append(
            stratify_by_percentile(image, percentile_ranges))

    LOGGER.info("Concatenating stratified images...")
    combined = ee.Image.cat(stratified_images)
    num_bands = num_images
    concatenate_expression = " + ".join(
        [f"(b({i})*(100**{i}))" for i in range(num_bands)])
    concatenated = combined.expression(concatenate_expression).toInt()

    LOGGER.info("Requesting stratified sampling...")
    return concatenated.stratifiedSample(
        num_points,
        region=area_of_interest,
        scale=scale,
        geometries=True)


def download_features(features: ee.FeatureCollection) -> gpd.GeoDataFrame:
    LOGGER.info("Converting to GeoDataFrame...")
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
        start_date: datetime,
        end_date: datetime,
        method: Literal["convering_grid", "random", "stratified"],
        geo_file_path: str | None,
        num_points: int | None,
        num_strata: int,
        strat_scale: int,
        edge_size: int | float,
        meta_data_path) -> None:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    LOGGER.info("Loading geo file into GeoDataFrame...")
    gdf = gpd.read_file(geo_file_path)
    unary_polygon = unary_union(gdf["geometry"])
    gee_polygon = ee.Geometry.Polygon(
        list(unary_polygon.exterior.coords))
    # TODO: partition a polygon into smaller polygons and request samples in parallel

    LOGGER.info(f"Generating squares via {method}...")
    match method:
        case "convering_grid":
            squares = gee_polygon.coveringGrid(
                scale=edge_size)
        case "random":
            points = generate_random_points(
                gee_polygon, edge_size, num_points)
            squares = points.map(
                lambda f: draw_bounding_square(f, edge_size))
        case "stratified":
            points = stratified_sampling(
                num_points,
                num_strata,
                start_date,
                end_date,
                gee_polygon,
                strat_scale)
            squares = points.map(
                lambda f: draw_bounding_square(f, edge_size))

    LOGGER.info("Downloading squares from GEE...")
    gdf = download_features(squares)

    LOGGER.info("Processing squares for zarr...")
    df_out = gdf["square"]\
        .explode()\
        .apply(pd.Series)\
        .add_prefix("square_")\
        .map(tuple)
    df_out["point"] = gdf["geometry"].apply(lambda p: p.coords[0])
    df_out["constant"] = gdf["constant"]

    LOGGER.info("Converting to xarray and saving...")
    xarr = df_out.to_xarray().astype([("x", float), ("y", float)])
    xarr.to_zarr(
        store=meta_data_path,
        mode="a")


def generate_time_samples(
        start_year: int,
        end_year: int,
        back_step: int,
        training_ratio: float,
        test_ratio: float,
        meta_data_path: str,
        training_samples_path: str,
        validate_samples_path: str,
        test_samples_path: str) -> None:
    meta_data = xr.open_zarr(meta_data_path)
    years = range(end_year, start_year + back_step, -1)
    df_years = pd.DataFrame(years, columns=["year"])

    LOGGER.info("Creating coodinate-time combination samples...")
    df_list = []
    for idx in range(meta_data.sizes["index"]):
        _, point_name, _, polygon_name = parse_meta_data(
            meta_data, idx)
        new_df = df_years.copy()
        new_df["point"] = point_name
        new_df["polygon"] = polygon_name
        df_list.append(new_df)
    df_out = pd.concat(df_list, axis=0)
    df_out.index = range(len(df_out))

    LOGGER.info(
        "Splitting sample data into training, validation, and test sets...")

    df_train, df_test = train_test_split(
        df_out, test_size=test_ratio, random_state=round(time.time()))
    df_train, df_validate = train_test_split(
        df_train, test_size=training_ratio, random_state=round(time.time()))

    LOGGER.info(f"Saving sample data to path...")
    df_list = [df_train, df_validate, df_test]
    paths = [training_samples_path, validate_samples_path, test_samples_path]
    for df, path in zip(df_list, paths):
        yarr = df.to_xarray()
        yarr.to_zarr(
            store=path,
            mode="a")


def main(**kwargs):
    from settings import SAMPLER as configs
    if (config_path := kwargs["config_path"]) is not None:
        with open(config_path, "r") as f:
            configs = configs | yaml.safe_load(f)

    global LOGGER
    LOGGER = get_logger(configs["log_path"], configs["log_name"])

    try:
        if kwargs["generate_squares"]:
            LOGGER.info("Generating squares...")
            start = time.time()
            square_config = {
                "start_date": configs["start_date"],
                "end_date": configs["end_date"],
                "method": configs["method"],
                "geo_file_path": configs["geo_file_path"],
                "num_points": configs["num_points"],
                "num_strata": configs["num_strata"],
                "strat_scale": configs["strat_scale"],
                "edge_size": configs["edge_size"],
                "meta_data_path": configs["meta_data_path"],
            }
            generate_squares(**square_config)
            end = time.time()
            LOGGER.info(
                f"Generating squares completed in: {(end - start)/60:.2} minutes")

        if kwargs["generate_time_samples"]:
            LOGGER.info("Generating time samples...")
            start = time.time()
            time_samples_config = {
                "start_year": configs["start_date"].year,
                "end_year": configs["end_date"].year,
                "back_step": configs["back_step"],
                "training_ratio": configs["training_ratio"],
                "test_ratio": configs["test_ratio"],
                "meta_data_path": configs["meta_data_path"],
                "training_samples_path": configs["training_samples_path"],
                "validate_samples_path": configs["validate_samples_path"],
                "test_samples_path": configs["test_samples_path"],
            }
            generate_time_samples(**time_samples_config)
            end = time.time()
            LOGGER.info(
                f"Generating time samples completed in: {(end - start)/60:.2} minutes")
            # TODO:  develop a decorator for logging performance time
    except Exception as e:
        LOGGER.critical(f"Failed to generate samples: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Sampler Arguments')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--generate_squares', type=bool, default=True)
    parser.add_argument('--generate_time_samples', type=bool, default=True)
    return vars(parser.parse_args())


if __name__ == '__main__':
    main(**parse_args())
