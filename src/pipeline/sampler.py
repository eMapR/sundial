import ee
import geopandas as gpd
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import xarray as xr
import yaml

from datetime import datetime
from typing import Literal
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
from typing import Optional

from .downloader import Downloader
from .logger import get_logger
from .settings import SAMPLER, RANDOM_STATE, STRATA_ATTR_NAME
from .utils import parse_meta_data, lt_image_generator, zarr_reshape, function_timer, clip_xy_xarray, pad_xy_xarray

LOGGER = get_logger(SAMPLER["log_path"], os.getenv("SUNDIAL_METHOD"))


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

    # Getting mean images for stratified sampling
    raw_images = []
    raw_images.append(get_prism_image(area_of_interest, start_date, end_date))
    raw_images.append(get_elevation_image(area_of_interest))

    # stratify by percentile
    stratified_images = []
    for image in raw_images:
        percentile_ranges = get_percentile_ranges(
            image, area_of_interest, percentiles)
        stratified_images.append(
            stratify_by_percentile(image, percentile_ranges))

    # concatenate stratified images
    combined = ee.Image.cat(stratified_images)
    num_bands = num_images
    concatenate_expression = " + ".join(
        [f"(b({i})*(100**{i}))" for i in range(num_bands)])
    concatenated = combined.expression(concatenate_expression).toInt()

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


@function_timer(logger=LOGGER)
def generate_centroid_squares(
        geo_dataframe: gpd.GeoDataFrame,
        meter_edge_size: int) -> gpd.GeoDataFrame:
    geo_dataframe.loc[:, "geometry"] = geo_dataframe.loc[:, "geometry"]\
        .apply(lambda p: p.centroid.buffer(meter_edge_size//2).envelope)
    return geo_dataframe


@function_timer(logger=LOGGER)
def generate_square_samples(
        geo_dataframe: gpd.GeoDataFrame,
        method: Literal["convering_grid", "random", "gee_stratified", "centroid"],
        meta_data_path: str,
        meter_edge_size: int | float,
        fraction: Optional[float] = None,
        num_points: Optional[int] = None,
        gee_num_points: Optional[int] = None,
        gee_num_strata: Optional[int] = None,
        gee_start_date: Optional[datetime] = None,
        gee_end_date: Optional[datetime] = None,
        gee_strata_scale: Optional[int] = None,
        strata_map_path: Optional[str] = None,
        strata_columns: Optional[list[str]] = None) -> tuple[gpd.GeoDataFrame, int]:
    if fraction is not None:
        groupby = geo_dataframe.groupby(strata_columns)
        sample = groupby.sample(frac=fraction)
    elif num_points is not None:
        groupby = geo_dataframe.groupby(strata_columns)
        sample = groupby.sample(n=num_points)
    else:
        sample = geo_dataframe
    sample = sample.reset_index(drop=True)

    LOGGER.info(f"Generating squares from sample points using {method}...")
    match method:
        case "convering_grid":
            raise NotImplementedError
        case "random":
            raise NotImplementedError
        case "gee_stratified":
            raise NotImplementedError
        case "centroid":
            gdf = generate_centroid_squares(
                sample,
                meter_edge_size)

    LOGGER.info("Saving geo dataframe and strata to file...")
    if strata_columns is not None and strata_map_path is not None:
        gdf.loc[:, STRATA_ATTR_NAME] = gdf.loc[:, strata_columns].astype(
            str).apply(lambda x: '__'.join(x).replace(" ", "_"), axis=1)
        strata_map = {v: k for k, v in enumerate(
            gdf[STRATA_ATTR_NAME].unique())}
        with open(strata_map_path, "w") as f:
            yaml.dump(strata_map, f, default_flow_style=False)
    gdf.to_file(meta_data_path)

    return len(gdf), gdf


@function_timer(logger=LOGGER)
def generate_time_combinations(
        num_samples: int,
        start_year: int,
        end_year: int,
        back_step: int,
        time_sample_path: str) -> int:
    years = range(end_year, start_year + back_step, -1)
    df_years = pd.DataFrame(years, columns=["year"])

    df_list = []
    for idx in range(num_samples):
        new_df = df_years.copy()
        new_df.loc[:, "index"] = idx
        df_list.append(new_df)
    df_time = pd.concat(df_list, axis=0)
    df_time = df_time.reset_index(drop=True)
    df_time.to_file(time_sample_path)

    return len(df_time)


def rasterizer(polygons: gpd.GeoSeries,
               square: Polygon,
               scale: int):
    cols = int((square.bounds[2] - square.bounds[0]) / scale)
    rows = int((square.bounds[3] - square.bounds[1]) / scale)

    transform = from_bounds(*square.bounds, rows, cols)
    raster = rasterize(
        shapes=polygons,
        out_shape=(rows, cols),
        fill=0,
        transform=transform)

    return raster


def rasterizer_helper(population_gdf: gpd.GeoDataFrame,
                      sample_gdf: gpd.GeoDataFrame,
                      strata_columns: list[str | int],
                      groupby_columns: list[str | int],
                      strata_map: dict[str, int],
                      pixel_edge_size: int,
                      scale: int,
                      anno_data_path: str,
                      io_limit: int,
                      io_lock: Optional[mp.Lock] = None,
                      index_queue: Optional[mp.Queue] = None):
    columns = groupby_columns + strata_columns
    batch = []
    while (index := index_queue.get()) is not None:
        LOGGER.info(f"Rasterizing sample {index}...")
        target = sample_gdf.iloc[index]
        square = target.geometry
        group = target[columns]
        stratum_idx = strata_map[target[STRATA_ATTR_NAME]]
        mp = population_gdf[(population_gdf[columns] == group)
                            .all(axis=1)].geometry
        annotation = rasterizer(mp, square, scale)
        annotation = xr.DataArray(annotation, dims=["y", "x"])

        if pixel_edge_size > min(annotation["x"].size,  annotation["y"].size):
            annotation = pad_xy_xarray(annotation, pixel_edge_size)
        if pixel_edge_size < max(annotation["x"].size,  annotation["y"].size):
            annotation = clip_xy_xarray(annotation, pixel_edge_size)

        xarr_anno_list = [xr.DataArray(np.zeros(annotation.shape), dims=['x', 'y'])
                          for _ in strata_map]
        xarr_anno = xr.concat(xarr_anno_list, dim=STRATA_ATTR_NAME)
        xarr_anno[stratum_idx:stratum_idx+1, :, :] = annotation
        xarr_anno.name = str(index)

        batch.append(xarr_anno)
        if len(batch) == io_limit:
            xarr_anno = xr.merge(batch)
            with io_lock:
                xarr_anno.to_zarr(store=anno_data_path, mode="a")
            batch.clear()

    xarr_anno = xr.merge(batch)
    xarr_anno.to_zarr(store=anno_data_path, mode="a")


@function_timer(logger=LOGGER)
def generate_annotation_chips(
        population_gdf: gpd.GeoDataFrame,
        sample_gdf: gpd.GeoDataFrame,
        strata_columns: list[str],
        groupby_columns: list[str],
        strata_map_path: str,
        pixel_edge_size: int,
        scale: int,
        anno_data_path: str,
        num_workers: int,
        io_limit: int,):
    num_samples = len(sample_gdf)
    with open(strata_map_path, "r") as f:
        strata_map = yaml.safe_load(f)

    manager = mp.Manager()
    index_queue = manager.Queue()
    io_lock = manager.Lock()

    [index_queue.put(i) for i in range(num_samples)]
    [index_queue.put(None) for _ in range(num_workers)]
    rasterizers = set()
    for _ in range(num_workers):
        p = mp.Process(
            target=rasterizer_helper,
            args=(population_gdf,
                  sample_gdf,
                  strata_columns,
                  groupby_columns,
                  strata_map,
                  pixel_edge_size,
                  scale,
                  anno_data_path,
                  io_limit,
                  io_lock,
                  index_queue),
            daemon=True)
        p.start()
        rasterizers.add(p)
    [p.join() for p in rasterizers]


@function_timer(logger=LOGGER)
def generate_image_chips(downloader_kwargs):
    downloader = Downloader(**downloader_kwargs)
    downloader.start()


def sample():
    num_samples = None
    try:
        LOGGER.info("Loading geo file into GeoDataFrame...")
        geo_dataframe = gpd.read_file(SAMPLER["geo_file_path"])

        if SAMPLER["generate_square_samples"]:
            LOGGER.info("Generating squares...")
            square_config = {
                "geo_dataframe": geo_dataframe,
                "method": SAMPLER["method"],
                "meta_data_path": SAMPLER["meta_data_path"],
                "meter_edge_size": SAMPLER["scale"] * SAMPLER["pixel_edge_size"],
                "fraction": SAMPLER["fraction"],
                "num_points": SAMPLER["num_points"],
                "gee_num_points": SAMPLER["num_points"],
                "gee_num_strata": SAMPLER["num_strata"],
                "gee_start_date": SAMPLER["start_date"],
                "gee_end_date": SAMPLER["end_date"],
                "gee_strata_scale": SAMPLER["strata_scale"],
                "strata_map_path": SAMPLER["strata_map_path"],
                "strata_columns": SAMPLER["strata_columns"],
            }
            num_samples, gdf = generate_square_samples(**square_config)

        if num_samples is None:
            gdf = gpd.read_file(SAMPLER["meta_data_path"])
            num_samples = len(gdf)

        if SAMPLER["generate_time_combinations"]:
            LOGGER.info("Generating time combinations...")
            time_sample_config = {
                "start_year": SAMPLER["start_date"].year,
                "end_year": SAMPLER["end_date"].year,
                "back_step": SAMPLER["back_step"],
                "time_sample_path": SAMPLER["time_sample_path"]
            }
            num_samples = generate_time_combinations(
                num_samples, **time_sample_config)

        if SAMPLER["generate_train_test_splits"]:
            LOGGER.info(
                "Splitting sample data into training, validation, test, and predict sets...")
            indices = np.arange(num_samples)
            train, validate = train_test_split(
                indices, test_size=SAMPLER["validate_ratio"])
            validate, test = train_test_split(
                validate, test_size=SAMPLER["test_ratio"])
            test, predict = train_test_split(
                test, test_size=SAMPLER["predict_ratio"])

            LOGGER.info("Saving sample data splits to paths...")
            idx_lsts = [train, test, validate, predict]
            paths = [SAMPLER["train_sample_path"],
                     SAMPLER["validate_sample_path"],
                     SAMPLER["test_sample_path"],
                     SAMPLER["predict_sample_path"]]
            for path, idx_lst in zip(paths, idx_lsts):
                np.save(path, idx_lst)
        else:
            if not os.path.exists(SAMPLER["train_sample_path"]):
                np.save(SAMPLER["train_sample_path"], np.arange(num_samples))

    except Exception as e:
        LOGGER.critical(f"Failed to generate sample: {type(e)} {e}")
        raise e


def annotate():
    LOGGER.info("Loading geo files into GeoDataFrame...")
    geo_dataframe = gpd.read_file(SAMPLER["geo_file_path"])
    gdf = gpd.read_file(SAMPLER["meta_data_path"])

    if SAMPLER["generate_annotation_data"]:
        LOGGER.info("Generating annotations from samples...")
        annotation_config = {
            "population_gdf": geo_dataframe,
            "sample_gdf": gdf,
            "strata_columns": SAMPLER["strata_columns"],
            "groupby_columns": SAMPLER["groupby_columns"],
            "strata_map_path": SAMPLER["strata_map_path"],
            "pixel_edge_size": SAMPLER["pixel_edge_size"],
            "scale": SAMPLER["scale"],
            "anno_data_path": SAMPLER["anno_data_path"],
            "num_workers": SAMPLER["num_workers"],
            "io_limit": SAMPLER["io_limit"],
        }
        generate_annotation_chips(**annotation_config)


def download():
    LOGGER.info("Loading geo file into GeoDataFrame...")
    gdf = gpd.read_file(SAMPLER["meta_data_path"])
    if SAMPLER["generate_image_chip_data"]:
        LOGGER.info("Generating image chips from samples...")
        downloader_kwargs = {
            "file_type": SAMPLER["file_type"],
            "overwrite": SAMPLER["overwrite"],
            "scale": SAMPLER["scale"],
            "pixel_edge_size": SAMPLER["pixel_edge_size"],
            "projection": SAMPLER["projection"],
            "look_years": SAMPLER["look_years"],
            "chip_data_path": SAMPLER["chip_data_path"],
            "meta_data": gdf,
            "meta_data_parser": parse_meta_data,
            "image_expr_generator": lt_image_generator,
            "image_reshaper": zarr_reshape,
            "num_workers": SAMPLER["num_workers"],
            "io_limit": SAMPLER["io_limit"],
            "logger": LOGGER,
        }
        generate_image_chips(downloader_kwargs)
