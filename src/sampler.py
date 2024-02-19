import ee
import argparse
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
from typing import Literal
from shapely.geometry import Polygon, MultiPolygon
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
    return sorted(single_band_image.reduceRegion(
        reducer=ee.Reducer.percentile(percentiles),
        geometry=area_of_interest,
        maxPixels=1e13
    ).values().getInfo())


def stratify_by_percentile(
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


def stratified_sampling(
        num_points_per: int,
        num_subclasses: int,
        start_date: datetime,
        end_date: datetime,
        area_of_interest: Polygon | MultiPolygon,
        scale: int) -> gpd.GeoDataFrame:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    area_of_interest = ee.Geometry.Polygon(
        list(area_of_interest.exterior.coords))
    percentiles = ee.List.sequence(0, 100, count=num_subclasses+1)

    LOGGER.info("Getting raw images for stratisfied sampling...")
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
    num_bands = len(stratified_images)
    concatenate_expression = " + ".join(
        [f"(b({i})*(100**{i}))" for i in range(num_bands)])
    concatenated = combined.expression(concatenate_expression).toInt()

    LOGGER.info("Sampling points...")
    features = concatenated.stratifiedSample(
        num_points_per,
        region=area_of_interest,
        scale=scale,
        geometries=True)

    LOGGER.info("Converting to GeoDataFrame...")
    # TODO: add try, exceptions, and logging
    try:
        return ee.data.computeFeatures({
            "expression": features,
            "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    except Exception as e:
        LOGGER.critical(
            f"Failed to convert feature collection to GeoDataFrame: {e}")
        raise e


def generate_overlapping_squares(
        polygon: Polygon | MultiPolygon,
        edge_size: float) -> list[Polygon]:

    x_min, y_min, x_max, y_max = polygon.bounds
    x_padding = (((x_max - x_min) % edge_size) / 2)
    y_padding = (((y_max - y_min) % edge_size) / 2)

    if edge_size > (y_max - y_min):
        y_padding = 0
    if edge_size > (x_max - x_min):
        x_padding = 0

    x_start = x_min - x_padding
    y_start = y_min - y_padding
    x_end = x_max + x_padding
    y_end = y_max + y_padding

    squares = []
    x = x_start
    while x < x_end:
        y = y_start
        while y < y_end:
            square = [(x, y),
                      (x + edge_size, y),
                      (x + edge_size, y + edge_size),
                      (x, y + edge_size),
                      (x, y)]
            if square.intersects(polygon):
                squares.append(square)
            y += edge_size
        x += edge_size

    return squares


def generate_overlapping_indices(
        square_pixel_length: int,
        index_pixel_length: int) -> list[list[list[int, int]]]:
    padding = square_pixel_length % index_pixel_length
    end_padding = padding // 2
    start_padding = padding - end_padding

    indices = []
    x = start_padding
    while x < square_pixel_length:
        y = start_padding
        while y < square_pixel_length:
            indices.append([[x, x + index_pixel_length],
                           [y, y + index_pixel_length]])
            y += index_pixel_length
        x += index_pixel_length
    return indices


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


def generate_gaussian_indices(
        square_pixel_length: int,
        index_pixel_length: int,
        std_dev: float,
        num_squares: int) -> list[list[list[int, int]]]:
    x_coords = np.random.normal(square_pixel_length // 2, std_dev, num_squares)
    y_coords = np.random.normal(square_pixel_length // 2, std_dev, num_squares)
    index_centroids = np.column_stack((x_coords, y_coords))
    start_diff = index_pixel_length // 2
    end_diff = start_diff
    if index_pixel_length % 2 == 0:
        start_diff -= 1

    indices = []
    for centroid in index_centroids:
        if all(
            [centroid[0] - start_diff >= 0,
             centroid[0] + end_diff < square_pixel_length,
             centroid[1] - start_diff >= 0,
             centroid[1] + end_diff < square_pixel_length]
        ):
            indices.append([[centroid[0] - start_diff, centroid[0] + end_diff],
                            [centroid[1] - start_diff, centroid[1] + end_diff]])

    return indices


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
        method: Literal["overlap", "gaussian", "single"],
        file_path: str | None,
        new_points: bool,
        num_points_per: int | None,
        num_subclasses: int,
        scale: int,
        save_path: str | None,
        edge_size: float,
        std_dev: float | None,
        num_squares: int | None,
        meta_data_path) -> None:

    LOGGER.info("Loading geo file into GeoDataFrame...")
    gdf = gpd.read_file(file_path)
    if new_points:
        LOGGER.info("Generating stratified samples into GeoDataFrame...")
        if num_points_per is None:
            raise ValueError("num_points is required for generating polygons")
        if num_subclasses is None:
            raise ValueError(
                "num_subclasses is required for generating polygons")
        if scale is None:
            raise ValueError("scale is required for generating polygons")
        gdf = stratified_sampling(
            num_points_per,
            num_subclasses,
            start_date,
            end_date,
            gdf["geometry"][0],
            scale)
        if save_path is not None:
            gdf.to_file(save_path)

    match method:
        case "overlap":
            LOGGER.info("Generating overlapping squares...")
            df_list = []
            for polygon in gdf["geometry"]:
                df = pd.DataFrame(
                    generate_overlapping_squares(polygon, edge_size),
                    columns=SQUARE_COLUMNS)
                df["point"] = [polygon.centroid.coords[0]] * df.shape[0]
                df_list.append(df)
            df_out = pd.concat(df_list, axis=0)

        case "gaussian":
            LOGGER.info("Generating gaussian squares...")
            if std_dev is None:
                raise ValueError("std_dev is required for gaussian sampling")
            if num_squares is None:
                raise ValueError(
                    "num_squares is required for gaussian sampling")
            df_list = []
            for point in gdf["geometry"]:
                point = point.centroid.coords[0]
                df = pd.DataFrame(
                    generate_gaussian_squares(
                        point,
                        edge_size,
                        std_dev,
                        num_squares),
                    columns=SQUARE_COLUMNS)
                df["point"] = [point] * df.shape[0]
                df_list.append(df)
                break
            df_out = pd.concat(df_list, axis=0)

        case "single":
            LOGGER.info("Generating single squares...")
            df_out = pd.DataFrame(
                [generate_single_square(point, edge_size)
                 for point in gdf["geometry"]],
                columns=SQUARE_COLUMNS)
        case _:
            LOGGER.critical("Invalid method")
            raise ValueError("Invalid method")

    LOGGER.info("Saving squares to file...")
    df_out.index = range(len(df_out))
    xarr = df.to_xarray().astype([("x", float), ("y", float)])
    xarr.to_zarr(
        store=meta_data_path,
        mode="a")


def generate_time_samples(
        start_year: int,
        end_year: int,
        back_step: int,
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
        "Splitting data into training, validation, and prediction sets...")
    df_train, df_validate = train_test_split(
        df_out, test_size=0.20, random_state=round(time.time()))
    df_validate, df_test = train_test_split(
        df_validate, test_size=0.01, random_state=round(time.time()))

    df_list = [df_train, df_validate, df_test]
    paths = [training_samples_path, validate_samples_path, test_samples_path]

    for df, path in zip(df_list, paths):
        yarr = df.to_xarray()
        yarr.to_zarr(
            store=path,
            mode="a")


def main(**kwargs):
    if (config_path := kwargs["config_path"]) is not None:
        with open(config_path, "r") as f:
            configs = yaml.safe_load(f)
    else:
        from settings import SAMPLER as configs
    global LOGGER
    LOGGER = get_logger(configs["log_path"], configs["log_name"])

    if kwargs["generate_squares"]:
        square_config = {
            "start_date": configs["start_date"],
            "end_date": configs["end_date"],
            "method": configs["method"],
            "file_path": configs["file_path"],
            "new_points": configs["new_points"],
            "num_points_per": configs["num_points_per"],
            "num_subclasses": configs["num_subclasses"],
            "scale": configs["scale"],
            "save_path": configs["save_path"],
            "edge_size": configs["edge_size"],
            "std_dev": configs["std_dev"],
            "num_squares": configs["num_squares"],
            "meta_data_path": configs["meta_data_path"],
        }
        generate_squares(**square_config)

    if kwargs["generate_time_samples"]:
        time_samples_config = {
            "start_year": configs["start_date"].year,
            "end_year": configs["end_date"].year,
            "back_step": configs["back_step"],
            "meta_data_path": configs["meta_data_path"],
            "training_samples_path": configs["training_samples_path"],
            "validate_samples_path": configs["validate_samples_path"],
            "test_samples_path": configs["test_samples_path"],
        }
        generate_time_samples(**time_samples_config)


def parse_args():
    parser = argparse.ArgumentParser(description='Sampler Arguments')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--generate_squares', type=bool, default=True)
    parser.add_argument('--generate_time_samples', type=bool, default=True)
    return vars(parser.parse_args())


if __name__ == '__main__':
    main(**parse_args())
