import ee
import argparse
import yaml

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon

from utils import parse_meta_data, SQUARE_COLUMNS


def load_geo_file(geo_file_path: str) -> gpd.GeoDataFrame:
    with open(geo_file_path) as f:
        data = gpd.read_file(f)
    return data


def get_elevation_image(area_of_interest: Polygon) -> ee.Image:
    area_of_interest = ee.Geometry.Polygon(
        list(area_of_interest.exterior.coords))
    return ee.Image('USGS/SRTMGL1_003').clip(area_of_interest)


def get_climate_image(area_of_interest: Polygon) -> ee.Image:
    area_of_interest = ee.Geometry.Polygon(
        list(area_of_interest.exterior.coords))
    return ee.Image("WORLDCLIM/V1/BIO").clip(area_of_interest)


def get_nlcd_image(area_of_interest: Polygon) -> ee.Image:
    area_of_interest = ee.Geometry.Polygon(
        list(area_of_interest.exterior.coords))
    return ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").first().clip(area_of_interest)


def get_prism_image(area_of_interest: Polygon) -> ee.Image:
    area_of_interest = ee.Geometry.Polygon(
        list(area_of_interest.exterior.coords))
    collection = ee.ImageCollection(
        "OREGONSTATE/PRISM/AN81m").clip(area_of_interest)
    return collection.reduce(ee.Reducer.mean())


def stratified_sampling(image_source: str, class_band: str, num_classes, int) -> gpd.GeoDataFrame:
    # TODO
    pass


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
    dist_from_centroid = edge_size / 2

    x_coords = np.random.normal(centroid_x, std_dev, num_squares)
    y_coords = np.random.normal(centroid_y, std_dev, num_squares)
    square_centroids = np.column_stack((x_coords, y_coords))
    squares = [generate_single_square(
        square_centroids, dist_from_centroid) for i in range(num_squares)]

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
    return [
        (centroid_coords[0] - dist_from_centroid,
            centroid_coords[1] + dist_from_centroid),
        (centroid_coords[0] + dist_from_centroid,
            centroid_coords[1] + dist_from_centroid),
        (centroid_coords[0] + dist_from_centroid,
            centroid_coords[1] - dist_from_centroid),
        (centroid_coords[0] - dist_from_centroid,
            centroid_coords[1] - dist_from_centroid),
        (centroid_coords[0] - dist_from_centroid,
            centroid_coords[1] + dist_from_centroid)
    ]


def generate_squares(
        method="gaussian",
        file_path=None,
        num_points=100,
        edge_size=.07,
        std_dev=10,
        num_squares=100,
        meta_data_path="meta_data.zarr") -> None:

    # get points from which to generate squares
    if file_path is not None:
        gdf = load_geo_file(file_path)
    else:
        if num_points is None:
            raise ValueError("num_points is required for generating polygons")
        gdf = stratified_sampling(num_points)

    # perform checks and generate squares
    match method:
        case "overlap":
            # TODO: Parallelize this
            df_list = []
            for polygon in gdf["geometry"]:
                df = pd.DataFrame(
                    generate_overlapping_squares(polygon, edge_size),
                    columns=SQUARE_COLUMNS)
                # TODO: save additional meta data
                df["point"] = [polygon.centroid.coords[0]] * df.shape[0]
                df_list.append(df)
            df_out = pd.concat(df_list, axis=0)

        case "gaussian":
            if num_squares is None:
                raise ValueError(
                    "num_squares is required for gaussian sampling")
            if std_dev is None:
                raise ValueError("std_dev is required for gaussian sampling")
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
                # TODO: save additional meta data
                df["point"] = [point] * df.shape[0]
                df_list.append(df)
            df_out = pd.concat(df_list, axis=0)

        case "single":
            df_out = pd.DataFrame(
                [generate_single_square(point, edge_size)
                 for point in gdf["geometry"]],
                columns=SQUARE_COLUMNS)
        case _:
            raise ValueError("Invalid method")

    # save generated data to file
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
        training_data_path: str,
        validate_data_path: str,
        predict_data_path: str) -> None:
    meta_data = xr.read_zarr(meta_data_path)
    years = range(end_year, start_year + back_step, -1)
    df_years = pd.DataFrame(years, columns=["year"])

    # generate combintations of points and years
    df_list = []
    for idx in range(meta_data.sizes["index"]):
        _, point_name, _, polygon_name = parse_meta_data(
            meta_data, idx)
        new_df = df_years.copy()
        new_df["point"] = point_name
        new_df["polygon"] = polygon_name
        df_list.append(new_df)

    # TODO: split combinations into training, validation, and prediction
    # save combinations to file
    df_out = pd.concat(df_list, axis=0)
    df_out.index = range(len(df_out))
    yarr = df_out.to_xarray().astype([("year", int)])
    yarr.to_zarr(
        store=training_data_path,
        mode="a")


def main(**kwargs):
    if (configs_path := kwargs["configs_path"]) is not None:
        with open(configs_path, "r") as f:
            configs = yaml.safe_load(f)
    else:
        from settings import SAMPLER as configs

    if kwargs["generate_squares"]:
        square_config = {
            "method": configs["method"],
            "file_path": configs["file_path"],
            "meta_data_path": configs["meta_data_path"],
            "num_points": configs["num_points"],
            "edge_size": configs["edge_size"],
            "std_dev": configs["std_dev"],
            "num_squares": configs["num_squares"],
        }
        generate_squares(**square_config)

    if kwargs["generate_time_samples"]:
        time_samples_config = {
            "start_year": configs["start_year"],
            "end_year": configs["end_year"],
            "back_step": configs["back_step"],
            "meta_data_path": configs["meta_data_path"],
            "training_data_path": configs["training_data_path"],
            "validate_data_path": configs["validate_data_path"],
            "predict_data_path": configs["predict_data_path"],
        }
        generate_time_samples(**time_samples_config)


def parse_args():
    parser = argparse.ArgumentParser(description='Sampler Arguments')
    parser.add_argument('--configs_path', type=str, default=None)
    parser.add_argument('--generate_squares', type=bool, default=True)
    parser.add_argument('--generate_time_samples', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(**parse_args())
