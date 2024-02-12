import ee
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon, MultiPolygon


def load_geojson(geojson_path: str) -> np.array:
    with open(geojson_path) as f:
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


def stratified_sampling(image_source: str, class_band: str) -> gpd.GeoDataFrame:
    # TODO
    pass


def generate_overlapping_squares(
        polygon: Point | Polygon | MultiPolygon,
        edge_size: float) -> tuple(MultiPolygon, int):

    x_min, y_min, x_max, y_max = polygon.bounds
    x_buff = (((x_max - x_min) % edge_size) / 2)
    y_buff = (((y_max - y_min) % edge_size) / 2)

    if edge_size > (y_max - y_min):
        y_buff = 0
    if edge_size > (x_max - x_min):
        x_buff = 0

    x_start = x_min - x_buff
    y_start = y_min - y_buff
    x_end = x_max + x_buff
    y_end = y_max + y_buff

    squares = []
    x = x_start
    while x < x_end:
        y = y_start
        while y < y_end:
            square = Polygon([(x, y),
                              (x + edge_size, y),
                              (x + edge_size, y + edge_size),
                              (x, y + edge_size),
                              (x, y)])
            if square.intersects(Polygon(polygon)):
                squares.append(square)
            y += edge_size
        x += edge_size

    return MultiPolygon(squares), len(squares)


def generate_gaussian_squares(
        polygon: Point | Polygon | MultiPolygon,
        edge_size: float,
        std_dev: float,
        num_squares: int) -> tuple(MultiPolygon, int):

    centroid_x, centroid_y = polygon.centroid
    dist_from_centroid = edge_size / 2
    x_coords = np.random.normal(centroid_x, std_dev, num_squares)
    y_coords = np.random.normal(centroid_y, std_dev, num_squares)
    square_centroids = np.column_stack((x_coords, y_coords))
    squares = [Polygon([
        (square_centroids[i][0] - dist_from_centroid,
            square_centroids[i][1] + dist_from_centroid)
        (square_centroids[i][0] + dist_from_centroid,
            square_centroids[i][1] + dist_from_centroid)
        (square_centroids[i][0] + dist_from_centroid,
            square_centroids[i][1] - dist_from_centroid)
        (square_centroids[i][0] - dist_from_centroid,
            square_centroids[i][1] - dist_from_centroid)
        (square_centroids[i][0] - dist_from_centroid,
            square_centroids[i][1] + dist_from_centroid)
    ]) for i in range(num_squares)]

    return MultiPolygon(squares), len(squares)


def main(**kwargs):
    edge_size = kwargs.get("edge_size")
    meta_data_path = kwargs.get("meta_data_path")

    if (geojson_path := kwargs.get("geojson_path")) is not None:
        gdf = load_geojson(geojson_path)
    else:
        if (num_points := kwargs.get("num_points")) is None:
            raise ValueError("num_points is required for generating polygons")
        gdf = stratified_sampling(num_points)

    match kwargs.get("method"):
        # TODO: Implement mp for iterating over the gdf
        case "overlap":
            squares_df = pd.DataFrame([
                generate_overlapping_squares(p, edge_size)
                for p in gdf["geometry"]],
                columns=["squares", "square_count"]
            )

        case "gaussian":
            if (num_squares := kwargs.get("num_squares")) is None:
                raise ValueError(
                    "num_squares is required for gaussian sampling")
            if (std_dev := kwargs.get("std_dev")) is None:
                raise ValueError("std_dev is required for gaussian sampling")
            squares_df = pd.DataFrame([
                generate_gaussian_squares(p, edge_size, std_dev, num_squares)
                for p in gdf["geometry"]],
                columns=["squares", "square_count"]
            )

    for column in squares_df.columns:
        gdf[column] = gdf[column]
    gdf.to_file(meta_data_path, driver='GeoJSON')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Gaussian Squares')
    parser.add_argument('--edge_size', type=int, required=True,
                        help='Edge size of the squares')
    parser.add_argument('--num_points', type=int,
                        help='Number of points to generate per coordinate')
    parser.add_argument('--num_squares', type=int,
                        help='Number of squares to generate per polygon')
    parser.add_argument('--std_dev', type=int,
                        help='Standard deviation for generating coordinates')
    parser.add_argument('--geojson_path', type=str,
                        help='Path to geojson file containing the polygons')
    parser.add_argument('--meta_data_path', type=str,
                        default="meta_data.geojson", help='Path to save the generated data')
    parser.add_argument('--method', type=str, default="overlap",
                        help='Method to sample from. Either "overlap" or "gaussian"')

    return parser.parse_args()


if __name__ == '__main__':
    main(**parse_args())
