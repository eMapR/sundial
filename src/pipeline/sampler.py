import ee
import geopandas as gpd
import multiprocessing as mp
import numpy as np
import os
import xarray as xr
import yaml

from datetime import datetime
from typing import Literal, Any
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely import unary_union
from sklearn.model_selection import train_test_split
from typing import Optional

from .downloader import Downloader
from .logger import get_logger
from .settings import (
    save_config,
    SAMPLER_CONFIG,
    RANDOM_STATE,
    META_DATA_PATH,
    STRATA_ATTR_NAME,
    NO_DATA_VALUE,
    CHIP_DATA_PATH,
    ANNO_DATA_PATH,
    TRAIN_SAMPLE_PATH,
    VALIDATE_SAMPLE_PATH,
    TEST_SAMPLE_PATH,
    PREDICT_SAMPLE_PATH,
    STRATA_MAP_PATH,
    STAT_DATA_PATH,
    GEO_PRE_PATH,
    GEO_RAW_PATH,
    LOG_PATH
)
from .utils import (
    parse_meta_data,
    lt_image_generator,
    zarr_reshape,
    function_timer,
    clip_xy_xarray,
    pad_xy_xarray,
    get_xarr_mean_std
)

LOGGER = get_logger(LOG_PATH, os.getenv("SUNDIAL_METHOD"))


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


@function_timer(logger=LOGGER)
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


@function_timer(logger=LOGGER)
def gee_generate_random_points(
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


@function_timer(logger=LOGGER)
def gee_stratified_sampling(
        num_points: int,
        num_strata: int,
        scale: int,
        start_date: datetime,
        end_date: datetime,
        sources: Literal["prism", "elevation", "ads_score"],
        area_of_interest: ee.Geometry,
        projection: str) -> ee.FeatureCollection:
    # creating percentiles for stratification
    num_images = len(sources)
    percentiles = ee.List.sequence(0, 100, count=num_strata+1)

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


@function_timer(logger=LOGGER)
def gee_download_features(
        features: ee.FeatureCollection) -> gpd.GeoDataFrame:
    try:
        return ee.data.computeFeatures({
            "expression": features,
            "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    except Exception as e:
        LOGGER.critical(
            f"Failed to convert feature collection to GeoDataFrame: {e}")
        raise e


@function_timer(logger=LOGGER)
def preprocess_data(
        geo_dataframe: gpd.GeoDataFrame,
        preprocess_actions: list[str, Literal[">", "<", "==", "!=", "replace"], Any]) -> gpd.GeoDataFrame:
    for preprocess in preprocess_actions:
        column = preprocess["column"]
        action = preprocess["action"]
        target = preprocess["targets"]
        match action:
            case ">":
                geo_dataframe = geo_dataframe.loc[geo_dataframe[column] > target]
            case "<":
                geo_dataframe = geo_dataframe.loc[geo_dataframe[column] < target]
            case "==":
                if isinstance(target, str) or isinstance(target, int):
                    geo_dataframe = geo_dataframe.loc[geo_dataframe[column] == target]
                elif isinstance(target, list):
                    geo_dataframe = geo_dataframe.loc[geo_dataframe[column].isin(
                        target)]
                else:
                    raise ValueError(
                        f"Invalid filter target type: {type(target)}")
            case "!=":
                if isinstance(target, str) or isinstance(target, int):
                    geo_dataframe = geo_dataframe.loc[geo_dataframe[column] != target]
                elif isinstance(target, list):
                    geo_dataframe = geo_dataframe.loc[~geo_dataframe[column].isin(
                        target)]
                else:
                    raise ValueError(
                        f"Invalid filter target type: {type(target)}")
            case "replace":
                ogs, new = target
                geo_dataframe.loc[:, column] = geo_dataframe\
                    .loc[:, column].replace(ogs, new)
            case _:
                raise ValueError(f"Invalid filter operator: {action}")
    return geo_dataframe


@function_timer(logger=LOGGER)
def stratify_data(
    geo_dataframe: gpd.GeoDataFrame,
    fraction: Optional[float] = None,
    num_points: Optional[int] = None,
    strata_columns: Optional[list[str]] = None,
):
    if fraction is not None:
        groupby = geo_dataframe.groupby(strata_columns)
        sample = groupby.sample(frac=fraction)
    elif num_points is not None:
        groupby = geo_dataframe.groupby(strata_columns)
        sample = groupby.sample(n=num_points)
    else:
        sample = geo_dataframe
    sample = sample.reset_index(drop=True)
    return sample


@function_timer(logger=LOGGER)
def generate_centroid_squares(
        geo_dataframe: gpd.GeoDataFrame,
        meter_edge_size: int) -> gpd.GeoDataFrame:
    geo_dataframe.loc[:, "geometry"] = geo_dataframe.loc[:, "geometry"]\
        .apply(lambda p: p.centroid.buffer(meter_edge_size//2).envelope)
    return geo_dataframe


@function_timer(logger=LOGGER)
def generate_squares(
        geo_dataframe: gpd.GeoDataFrame,
        method: Literal["convering_grid", "random", "gee_stratified", "centroid"],
        meta_data_path: str,
        meter_edge_size: int | float,
        projection: Optional[str] = None,
        gee_stratafied_config: Optional[dict] = None,
        strata_map_path: Optional[str] = None,
        strata_columns: Optional[list[str]] = None) -> tuple[gpd.GeoDataFrame, int]:
    epsg = f"EPSG:{geo_dataframe.crs.to_epsg()}"
    if projection is not None and epsg != projection:
        LOGGER.info(f"Reprojecting geo dataframe to {projection}...")
        geo_dataframe = geo_dataframe.to_crs(projection)
        epsg = projection

    LOGGER.info(f"Generating squares from sample points via {method}...")
    match method:
        case "convering_grid":
            raise NotImplementedError
        case "random":
            raise NotImplementedError
        case "gee_stratified":
            ee.Initialize()
            even_odd = True if epsg == "EPSG:4326" else False
            area_of_interest = unary_union(geo_dataframe.geometry)
            area_of_interest = ee.Geometry.Polygon(
                list(area_of_interest.exterior.coords), proj=epsg, evenOdd=even_odd)
            gee_stratafied_config["area_of_interest"] = area_of_interest
            gee_stratafied_config["projection"] = epsg

            points = gee_stratified_sampling(**gee_stratafied_config)
            points = gee_download_features(points)\
                .set_crs("EPSG:4326")\
                .to_crs(epsg)
            gdf = generate_centroid_squares(points, meter_edge_size)
            gdf.loc[:, "year"] = gee_stratafied_config["end_date"].year
        case "centroid":
            gdf = generate_centroid_squares(
                geo_dataframe,
                meter_edge_size)

    if strata_columns is not None and strata_map_path is not None:
        LOGGER.info(f"Saving strata map to {strata_map_path}...")
        gdf.loc[:, STRATA_ATTR_NAME] = gdf.loc[:, strata_columns].astype(
            str).apply(lambda x: '__'.join(x).replace(" ", "_"), axis=1)
        strata_map = {v: k for k, v in enumerate(
            gdf[STRATA_ATTR_NAME].unique(), 1)}
        with open(strata_map_path, "w") as f:
            yaml.dump(strata_map, f, default_flow_style=False)
    LOGGER.info(f"Saving geo dataframe to {meta_data_path}...")
    gdf.to_file(meta_data_path)

    return len(gdf), gdf


@function_timer(logger=LOGGER)
def generate_time_combinations(
        num_samples: int,
        num_years: int,
        year_step: int) -> np.ndarray:
    start = num_years % year_step
    sample_arr = np.arange(num_samples)
    year_arr = np.arange(num_years)[start::year_step]

    sample_arr, year_arr = np.meshgrid(sample_arr, year_arr)
    sample_arr = sample_arr.flatten()
    year_arr = year_arr.flatten()
    time_arr = np.column_stack((sample_arr, year_arr))

    return time_arr


def rasterizer(polygons: gpd.GeoSeries,
               square: Polygon,
               scale: int,
               fill: int | float,
               default_value: int | float):
    if len(polygons) == 0:
        raise ValueError(
            f"Check preprocessing actions. Polygon series has len 0...")
    cols = int((square.bounds[2] - square.bounds[0]) / scale)
    rows = int((square.bounds[3] - square.bounds[1]) / scale)

    transform = from_bounds(*square.bounds, rows, cols)
    raster = rasterize(
        shapes=polygons,
        out_shape=(rows, cols),
        fill=fill,
        transform=transform,
        default_value=default_value)

    return raster


def rasterizer_helper(population_gdf: gpd.GeoDataFrame,
                      sample_gdf: gpd.GeoDataFrame,
                      strata_columns: list[str | int],
                      groupby_columns: list[str | int],
                      strata_map: dict[str, int],
                      pixel_edge_size: int,
                      scale: int,
                      flat_annotations: bool,
                      anno_data_path: str,
                      io_limit: int,
                      io_lock: Optional[mp.Lock] = None,
                      index_queue: Optional[mp.Queue] = None):
    columns = groupby_columns + strata_columns
    batch = []
    while (index := index_queue.get()) is not None:

        # getting annotation information from sample
        LOGGER.info(f"Rasterizing sample {index}...")
        target = sample_gdf.iloc[index]
        square = target.geometry
        group = target[columns]
        stratum_idx = strata_map[target[STRATA_ATTR_NAME]]
        default_value = stratum_idx if flat_annotations else 1

        # rasterizing multipolygon and clipping to square
        try:
            mp = population_gdf[(population_gdf[columns] == group)
                                .all(axis=1)].geometry
            # TODO: rasterize and save multipolygon using gdal
            annotation = rasterizer(
                mp, square, scale, NO_DATA_VALUE, default_value)
        except Exception as e:
            raise e
        annotation = xr.DataArray(annotation, dims=["y", "x"])

        # clipping or padding to pixel_edge_size
        if pixel_edge_size > min(annotation["x"].size,  annotation["y"].size):
            annotation = pad_xy_xarray(annotation, pixel_edge_size)
        if pixel_edge_size < max(annotation["x"].size,  annotation["y"].size):
            annotation = clip_xy_xarray(annotation, pixel_edge_size)

        # expanding annotation to (C ,H, W) shape
        if flat_annotations:
            xarr_anno = annotation
            xarr_anno = xarr_anno.expand_dims(STRATA_ATTR_NAME)
        else:
            # TODO: omit concat logic and generate with 3D shape
            xarr_anno_list = [xr.DataArray(np.zeros(annotation.shape), dims=['x', 'y'])
                              for _ in strata_map]
            xarr_anno = xr.concat(xarr_anno_list, dim=STRATA_ATTR_NAME)
            xarr_anno[stratum_idx-1:stratum_idx, :, :] = annotation

        # writing in batches to avoid io bottleneck
        LOGGER.info(f"Appending rasterized sample {index} to batch...")
        xarr_anno.name = str(index)
        batch.append(xarr_anno)
        if len(batch) == io_limit:
            xarr_anno = xr.merge(batch)
            with io_lock:
                xarr_anno.to_zarr(store=anno_data_path, mode="a")
            batch.clear()

        LOGGER.info(f"Rasterize sample {index} completed.")

    # writing remaining batch
    with io_lock:
        xarr_anno = xr.merge(batch)
        xarr_anno.to_zarr(store=anno_data_path, mode="a")


@function_timer(logger=LOGGER)
def generate_annotation_data(
        population_gdf: gpd.GeoDataFrame,
        sample_gdf: gpd.GeoDataFrame,
        strata_columns: list[str],
        groupby_columns: list[str],
        strata_map_path: str,
        pixel_edge_size: int,
        scale: int,
        flat_annotations: bool,
        anno_data_path: str,
        num_workers: int,
        io_limit: int,):
    num_samples = len(sample_gdf)
    with open(strata_map_path, "r") as f:
        strata_map = yaml.safe_load(f)

    manager = mp.Manager()
    index_queue = manager.Queue()
    io_lock = manager.Lock()

    LOGGER.info(
        f"Starting parallel process for {num_samples} samples using {num_workers}...")
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
                  flat_annotations,
                  anno_data_path,
                  io_limit,
                  io_lock,
                  index_queue),
            daemon=True)
        p.start()
        rasterizers.add(p)
    [r.join() for r in rasterizers]


@function_timer(logger=LOGGER)
def generate_image_chip_data(downloader_kwargs):
    downloader = Downloader(**downloader_kwargs)
    downloader.start()


@function_timer(logger=LOGGER)
def sample():
    num_samples = None
    try:
        LOGGER.info("Loading geo file into GeoDataFrame...")
        geo_dataframe = gpd.read_file(GEO_RAW_PATH)

        if SAMPLER_CONFIG["sample_toggles"]["preprocess_data"]:
            LOGGER.info("Preprocessing data in geo file...")
            sample_config = {
                "geo_dataframe": geo_dataframe,
                "preprocess_actions": SAMPLER_CONFIG["preprocess_actions"],
            }
            geo_dataframe = preprocess_data(**sample_config)
            geo_dataframe.to_file(GEO_PRE_PATH)

        if SAMPLER_CONFIG["sample_toggles"]["stratify_data"]:
            LOGGER.info("Stratifying data in geo file...")
            group_config = {
                "geo_dataframe": geo_dataframe,
                "fraction": SAMPLER_CONFIG["fraction"],
                "num_points": SAMPLER_CONFIG["num_points"],
                "strata_columns": SAMPLER_CONFIG["strata_columns"],
            }
            geo_dataframe = stratify_data(**group_config)

        if SAMPLER_CONFIG["sample_toggles"]["generate_squares"]:
            LOGGER.info("Generating squares...")
            square_config = {
                "geo_dataframe": geo_dataframe,
                "method": SAMPLER_CONFIG["method"],
                "meta_data_path": META_DATA_PATH,
                "meter_edge_size": SAMPLER_CONFIG["scale"] * SAMPLER_CONFIG["pixel_edge_size"],
                "projection": SAMPLER_CONFIG["projection"],
                "gee_stratafied_config": SAMPLER_CONFIG["gee_stratafied_config"],
                "strata_map_path": STRATA_MAP_PATH,
                "strata_columns": SAMPLER_CONFIG["strata_columns"],
            }
            num_samples, gdf = generate_squares(**square_config)

        if num_samples is None:
            gdf = gpd.read_file(META_DATA_PATH)
            num_samples = len(gdf)

        if SAMPLER_CONFIG["sample_toggles"]["generate_time_combinations"]:
            LOGGER.info("Generating time combinations...")
            num_years = SAMPLER_CONFIG["look_years"] + 1
            time_sample_config = {
                "num_samples": num_samples,
                "num_years": num_years,
                "year_step": SAMPLER_CONFIG["year_step"]
            }
            samples = generate_time_combinations(**time_sample_config)
        else:
            samples = np.arange(num_samples)

        if SAMPLER_CONFIG["sample_toggles"]["generate_train_test_splits"]:
            LOGGER.info(
                "Splitting sample data into training, validation, test, and predict sets...")
            train, validate = train_test_split(
                samples, test_size=SAMPLER_CONFIG["validate_ratio"])
            validate, test = train_test_split(
                validate, test_size=SAMPLER_CONFIG["test_ratio"])
            test, predict = train_test_split(
                test, test_size=SAMPLER_CONFIG["predict_ratio"])

            LOGGER.info("Saving sample data splits to paths...")
            idx_lsts = [train, validate, test, predict]
            paths = [TRAIN_SAMPLE_PATH,
                     VALIDATE_SAMPLE_PATH,
                     TEST_SAMPLE_PATH,
                     PREDICT_SAMPLE_PATH]
            for path, idx_lst in zip(paths, idx_lsts):
                np.save(path, idx_lst)
        else:
            if not os.path.exists(TRAIN_SAMPLE_PATH):
                np.save(TRAIN_SAMPLE_PATH, samples)

    except Exception as e:
        LOGGER.critical(f"Failed to generate sample: {type(e)} {e}")
        raise e


def annotate():
    try:
        LOGGER.info("Loading geo files into GeoDataFrame...")
        if os.path.exists(GEO_PRE_PATH):
            population_gdf = gpd.read_file(GEO_PRE_PATH)
        else:
            population_gdf = gpd.read_file(GEO_RAW_PATH)
        sample_gdf = gpd.read_file(META_DATA_PATH)

        LOGGER.info("Generating annotations...")
        annotation_config = {
            "population_gdf": population_gdf,
            "sample_gdf": sample_gdf,
            "strata_columns": SAMPLER_CONFIG["strata_columns"],
            "groupby_columns": SAMPLER_CONFIG["groupby_columns"],
            "strata_map_path": STRATA_MAP_PATH,
            "pixel_edge_size": SAMPLER_CONFIG["pixel_edge_size"],
            "scale": SAMPLER_CONFIG["scale"],
            "flat_annotations": SAMPLER_CONFIG["flat_annotations"],
            "anno_data_path": ANNO_DATA_PATH,
            "num_workers": SAMPLER_CONFIG["num_workers"],
            "io_limit": SAMPLER_CONFIG["io_limit"],
        }
        generate_annotation_data(**annotation_config)
    except Exception as e:
        LOGGER.critical(f"Failed to generate annotations: {type(e)} {e}")
        raise e


def download():
    try:
        LOGGER.info("Loading geo file into GeoDataFrame...")
        gdf = gpd.read_file(META_DATA_PATH)

        LOGGER.info("Generating image chips...")
        parser_kwargs = SAMPLER_CONFIG["medoid_config"]
        parser_kwargs["look_years"] = SAMPLER_CONFIG["look_years"]
        downloader_kwargs = {
            "file_type": SAMPLER_CONFIG["file_type"],
            "overwrite": SAMPLER_CONFIG["overwrite"],
            "scale": SAMPLER_CONFIG["scale"],
            "pixel_edge_size": SAMPLER_CONFIG["pixel_edge_size"],
            "projection": SAMPLER_CONFIG["projection"],
            "chip_data_path": CHIP_DATA_PATH,
            "meta_data": gdf,
            "meta_data_parser": parse_meta_data,
            "image_expr_generator": lt_image_generator,
            "image_reshaper": zarr_reshape,
            "num_workers": SAMPLER_CONFIG["gee_workers"],
            "io_limit": SAMPLER_CONFIG["io_limit"],
            "logger": LOGGER,
            "parser_kwargs": parser_kwargs,
        }
        generate_image_chip_data(downloader_kwargs)
    except Exception as e:
        LOGGER.critical(f"Failed to download images: {type(e)} {e}")
        raise e


@function_timer(logger=LOGGER)
def calculate():
    try:
        LOGGER.info("Calculating sample statistics...")
        match SAMPLER_CONFIG["file_type"]:
            case "ZARR":
                means, stds = get_xarr_mean_std(CHIP_DATA_PATH)
            case _:
                raise ValueError(
                    f"Invalid file type: {SAMPLER_CONFIG['file_type']}")
        save_config(
            {
                "means": means,
                "stds": stds
            },
            STAT_DATA_PATH
        )
    except Exception as e:
        LOGGER.critical(
            f"Failed to calculate sample statistics: {type(e)} {e}")
        raise e
