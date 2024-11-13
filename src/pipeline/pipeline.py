import ee
import geopandas as gpd
import importlib
import multiprocessing as mp
import numpy as np
import operator
import os
import pandas as pd
import random
import xarray as xr

from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely import unary_union
from typing import Any, Optional, Literal

from pipeline.downloader import Downloader
from pipeline.logging import function_timer, get_logger
from pipeline.settings import (
    DATETIME_LABEL,
    IDX_NAME_ZFILL,
    METHOD,
    NO_DATA_VALUE,
    RANDOM_SEED,
    CLASS_LABEL,
)
from pipeline.settings import (
    ALL_SAMPLE_PATH,
    ANNO_DATA_PATH,
    CHIP_DATA_PATH,
    GEO_POP_PATH,
    GEO_RAW_PATH,
    LOG_PATH,
    META_DATA_PATH,
    PREDICT_SAMPLE_PATH,
    SAMPLER_CONFIG,
    STAT_DATA_PATH,
    TEST_SAMPLE_PATH,
    TRAIN_SAMPLE_PATH,
    VALIDATE_SAMPLE_PATH,
)
from pipeline.utils import (
    gee_stratified_sampling,
    gee_download_features,
    generate_centroid_squares,
    get_class_weights,
    get_band_stats,
    get_xarr_stats,
    stratified_sample,
    train_validate_test_split,
    update_yaml,
)

LOGGER = get_logger(LOG_PATH, METHOD)
OPS_MAP = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


@function_timer
def preprocess_data(
        geo_dataframe: gpd.GeoDataFrame,
        preprocess_actions: list[dict[
            "column": str,
            "action": Literal[">", "<", "==", "!=", "replace"],
            "targets": int, float, str, list]],
        projection: Optional[str] = None,
        class_columns: Optional[list[str]] = None,
        datetime_column: Optional[str] = None) -> gpd.GeoDataFrame:
    epsg = f"EPSG:{geo_dataframe.crs.to_epsg()}"
    if projection is not None and epsg != projection:
        LOGGER.info(f"Reprojecting geo dataframe to {projection}...")
        geo_dataframe = geo_dataframe.to_crs(projection)
        epsg = projection

    mask = pd.Series(True, index=geo_dataframe.index)
    for preprocess in preprocess_actions:
        column = preprocess.get("column")
        action = preprocess.get("action")
        target = preprocess.get("targets")
        if action in OPS_MAP:
            mask &= OPS_MAP[action](geo_dataframe[column], target)
        else:
            match action:
                case "in":
                    assert isinstance(target, list)
                    mask &= geo_dataframe[column].isin(target)
                case "notin":
                    assert isinstance(target, list)
                    mask &= ~(geo_dataframe[column].isin(target))
                case "replace":
                    geo_dataframe.loc[:, column] = geo_dataframe[column].replace(
                        *target)
                case "unique":
                    mask &= ~(geo_dataframe[column].duplicated(keep=False))
                case _:
                    raise ValueError(f"Invalid filter operator: {action}")

    geo_dataframe = geo_dataframe[mask].copy()
    if datetime_column is not None:
        geo_dataframe.loc[:, DATETIME_LABEL] = geo_dataframe[datetime_column]

    if class_columns is not None:
        geo_dataframe.loc[:, CLASS_LABEL] = geo_dataframe[class_columns]\
            .apply(lambda x: '__'.join(x.astype(str)).replace(" ", "_"), axis=1)

    return geo_dataframe


@function_timer
def postprocess_data(
    chip_data: xr.Dataset,
    anno_data: xr.Dataset | None,
    postprocess_actions: list[dict[
        "data": Literal["chip", "anno"],
        "action": Literal["sum"],
        "operator": Literal[">", "<", ">=", "<=", "==", "!="],
        "targets": int, float, str, list]]):
    pre_calc = {"chip": {}, "anno": {}}
    data_map = {"chip": chip_data, "anno": anno_data}

    for postprocess in postprocess_actions:
        data = postprocess["data"]
        action = postprocess["action"]
        compare = OPS_MAP[postprocess["operator"]]
        target = postprocess["targets"]
        match action:
            case "sum":
                sums_ds = pre_calc[data].setdefault(action, data_map[data].sum())
                to_drop = [v for v in sums_ds.data_vars if not compare(sums_ds[v].values, target)]
                data_map[data] = data_map[data].drop_vars(to_drop, errors="ignore")
            case "filter":
                pass
            case _:
                raise ValueError(f"Invalid action: {action}")
    chip_vars = np.array(data_map["chip"].data_vars, dtype=int)
    anno_vars = np.array(data_map["anno"].data_vars, dtype=int)
    return np.intersect1d(chip_vars, anno_vars)


@function_timer
def generate_squares(
        geo_dataframe: gpd.GeoDataFrame,
        method: Literal["convering_grid", "random", "gee_stratified", "centroid"],
        meter_edge_size: int | float,
        squares_config: dict) -> gpd.GeoDataFrame:
    LOGGER.info(f"Generating squares from sample points via {method}...")
    match method:
        case "convering_grid":
            raise NotImplementedError
        case "random":
            raise NotImplementedError
        case "gee_stratified":
            ee.Initialize()
            even_odd = True if epsg == "EPSG:4326" else False
            epsg = f"EPSG:{geo_dataframe.crs.to_epsg()}"
            area_of_interest = unary_union(geo_dataframe.geometry)
            area_of_interest = ee.Geometry.Polygon(
                list(area_of_interest.exterior.coords), proj=epsg, evenOdd=even_odd)
            squares_config["area_of_interest"] = area_of_interest
            squares_config["projection"] = epsg

            points = gee_stratified_sampling(**squares_config)
            points = gee_download_features(points)\
                .set_crs("EPSG:4326")\
                .to_crs(epsg)
            gdf = generate_centroid_squares(points, meter_edge_size)
            gdf.loc[:, DATETIME_LABEL] = squares_config["end_date"]
        case "centroid":
            gdf = generate_centroid_squares(
                geo_dataframe,
                meter_edge_size
            )
        case _:
            raise ValueError(f"Invalid method: {method}")

    return gdf


@function_timer
def generate_time_combinations(
        samples: np.array,
        look_range: int,
        time_step: int) -> np.ndarray:
    start = look_range % time_step
    time_arr = np.arange(look_range)[start::time_step]

    sample_arr, time_arr = np.meshgrid(samples, time_arr)
    sample_arr = sample_arr.flatten()
    time_arr = time_arr.flatten()
    time_arr = np.column_stack((sample_arr, time_arr))

    return time_arr


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


def annotator(population_gdf: gpd.GeoDataFrame,
              sample_gdf: gpd.GeoDataFrame,
              groupby_columns: list[str | int],
              class_indices: dict[str, int],
              pixel_edge_size: int,
              scale: int,
              anno_data_path: str,
              io_limit: int,
              io_lock: Any = None,
              index_queue: Optional[mp.Queue] = None):
    batch = []
    while (index := index_queue.get()) is not None:
        index_name = str(index).zfill(IDX_NAME_ZFILL)
        
        # getting annotation information from sample
        LOGGER.info(f"Rasterizing sample {index_name}...")
        target = sample_gdf.iloc[index]
        group = target[groupby_columns]
        square = target.geometry

        # rasterizing multipolygon and clipping to square
        xarr_anno_list = []
        LOGGER.info(
            f"Creating annotations for sample {index_name} from class list...")

        # class list should already be in order of index value
        for default_value, class_name in zip(range(1, len(class_indices) + 1), class_indices):
            try:
                mask = (population_gdf[groupby_columns] == group).all(axis=1) & \
                    (population_gdf[CLASS_LABEL] == class_name)
                mp = population_gdf[mask].geometry
                if len(mp) == 0:
                    annotation = np.zeros((pixel_edge_size, pixel_edge_size))
                else:
                    annotation = rasterizer(
                        mp, square, pixel_edge_size, NO_DATA_VALUE, default_value)
            except Exception as e:
                raise e
            # TODO: add support for tif
            annotation = xr.DataArray(annotation, dims=["y", "x"])

            xarr_anno_list.append(annotation)
        xarr_anno = xr.concat(xarr_anno_list, dim=CLASS_LABEL)

        # writing in batches to avoid io bottleneck
        LOGGER.info(f"Appending rasterized sample {index_name} of shape {xarr_anno.shape} to batch...")
        xarr_anno.name = index_name
        batch.append(xarr_anno)
        if len(batch) == io_limit:
            xarr_anno = xr.merge(batch)
            with io_lock:
                xarr_anno.to_zarr(store=anno_data_path, mode="a")
            batch.clear()
            indices = list(xarr_anno.data_vars)
            LOGGER.info(f"Rasterized sample batch submitted... {indices}")

    # writing remaining batch
    if len(batch) > 0:
        with io_lock:
            xarr_anno = xr.merge(batch)
            xarr_anno.to_zarr(store=anno_data_path, mode="a")


@function_timer
def generate_annotation_data(
        population_gdf: gpd.GeoDataFrame,
        sample_gdf: gpd.GeoDataFrame,
        groupby_columns: list[str],
        pixel_edge_size: int,
        scale: int,
        anno_data_path: str,
        num_workers: int,
        io_limit: int,):
    num_samples = len(sample_gdf)
    class_indices = sorted(sample_gdf[CLASS_LABEL].unique())

    manager = mp.Manager()
    index_queue = manager.Queue()
    io_lock = manager.Lock()

    LOGGER.info(
        f"Starting parallel process for {num_samples} samples using {num_workers}...")
    [index_queue.put(i) for i in range(num_samples)]
    [index_queue.put(None) for _ in range(num_workers)]
    annotators = set()
    for _ in range(num_workers):
        p = mp.Process(
            target=annotator,
            args=(population_gdf,
                  sample_gdf,
                  groupby_columns,
                  class_indices,
                  pixel_edge_size,
                  scale,
                  anno_data_path,
                  io_limit,
                  io_lock,
                  index_queue),
            daemon=True)
        p.start()
        annotators.add(p)
    [r.join() for r in annotators]


@function_timer
def generate_image_chip_data(downloader_kwargs: dict):
    downloader = Downloader(**downloader_kwargs)
    downloader.start()


@function_timer
def sample():
    try:
        LOGGER.info("Loading geo file into GeoDataFrame...")
        geo_dataframe = gpd.read_file(GEO_RAW_PATH)

        if SAMPLER_CONFIG["preprocess_actions"]:
            LOGGER.info("Preprocessing data in geo file...")
            sample_config = {
                "geo_dataframe": geo_dataframe,
                "preprocess_actions": SAMPLER_CONFIG["preprocess_actions"],
                "projection": SAMPLER_CONFIG["projection"],
                "class_columns": SAMPLER_CONFIG["class_columns"],
                "datetime_column": SAMPLER_CONFIG["datetime_column"],
            }
            geo_dataframe = preprocess_data(**sample_config)
            LOGGER.info(f"Saving processed geo dataframe to {GEO_POP_PATH}...")
            geo_dataframe.to_file(GEO_POP_PATH)

        if SAMPLER_CONFIG["num_points"]:
            LOGGER.info("Performing stratified sample of data in geo file...")
            group_config = {
                "geo_dataframe": geo_dataframe,
                "class_label": CLASS_LABEL,
                "num_points": SAMPLER_CONFIG["num_points"],
            }
            geo_dataframe = stratified_sample(**group_config)

        LOGGER.info("Generating square polygons...")
        square_config = {
            "geo_dataframe": geo_dataframe,
            "method": SAMPLER_CONFIG["method"],
            "meter_edge_size": SAMPLER_CONFIG["scale"] * (SAMPLER_CONFIG["pixel_edge_size"]),    
            "squares_config": SAMPLER_CONFIG["squares_config"],
        }
        geo_dataframe = generate_squares(**square_config)

        LOGGER.info(f"Saving squares geo dataframe to {META_DATA_PATH}...")
        geo_dataframe.to_file(META_DATA_PATH)

    except Exception as e:
        LOGGER.critical(f"Failed to generate sample: {type(e)} {e}")
        raise e


@function_timer
def annotate():
    try:
        LOGGER.info("Loading geo files into GeoDataFrame...")
        if SAMPLER_CONFIG["preprocess_actions"] and os.path.exists(GEO_POP_PATH):
            population_gdf = gpd.read_file(GEO_POP_PATH)
        else:
            population_gdf = gpd.read_file(GEO_RAW_PATH)
        sample_gdf = gpd.read_file(META_DATA_PATH)

        LOGGER.info("Generating annotations...")
        annotation_config = {
            "population_gdf": population_gdf,
            "sample_gdf": sample_gdf,
            "groupby_columns": SAMPLER_CONFIG["groupby_columns"],
            "pixel_edge_size": SAMPLER_CONFIG["pixel_edge_size"],
            "scale": SAMPLER_CONFIG["scale"],
            "anno_data_path": ANNO_DATA_PATH,
            "num_workers": SAMPLER_CONFIG["num_workers"],
            "io_limit": SAMPLER_CONFIG["io_limit"],
        }
        generate_annotation_data(**annotation_config)
    except Exception as e:
        LOGGER.critical(f"Failed to generate annotations: {type(e)} {e}")
        raise e


@function_timer
def download():
    try:
        LOGGER.info("Loading meta_data into GeoDataFrame...")
        gdf = gpd.read_file(META_DATA_PATH)

        LOGGER.info("Generating image chips...")
        downloader_kwargs = {
            "file_type": SAMPLER_CONFIG["file_type"],
            "overwrite": SAMPLER_CONFIG["overwrite"],
            "scale": SAMPLER_CONFIG["scale"],
            "pixel_edge_size": SAMPLER_CONFIG["pixel_edge_size"],
            "projection": SAMPLER_CONFIG["projection"],
            "chip_data_path": CHIP_DATA_PATH,
            "meta_data": gdf,
            "meta_data_parser": getattr(importlib.import_module("pipeline.meta_data_parsers"), SAMPLER_CONFIG["meta_data_parser"]),
            "ee_image_factory": getattr(importlib.import_module("pipeline.ee_image_factories"), SAMPLER_CONFIG["ee_image_factory"]),
            "image_reshaper": getattr(importlib.import_module("pipeline.image_reshapers"), SAMPLER_CONFIG["image_reshaper"]),
            "num_workers": SAMPLER_CONFIG["gee_workers"],
            "io_limit": SAMPLER_CONFIG["io_limit"],
            "logger": LOGGER,
            "parser_kwargs": SAMPLER_CONFIG["parser_kwargs"],
            "factory_kwargs": SAMPLER_CONFIG["factory_kwargs"],
            "reshaper_kwargs": SAMPLER_CONFIG["reshaper_kwargs"],
        }
        generate_image_chip_data(downloader_kwargs)
    except Exception as e:
        LOGGER.critical(f"Failed to download images: {type(e)} {e}")
        raise e


@function_timer
def calculate():
    try:
        LOGGER.info("Calculating sample statistics...")
        stat = {}
        
        gdf = gpd.read_file(META_DATA_PATH)
        stat["class_count"] = gdf.groupby(CLASS_LABEL).size().to_dict()
        
        match SAMPLER_CONFIG["file_type"]:
            case "ZARR":
                LOGGER.info(f"Verifying chip data...")
                chip_data = xr.open_zarr(CHIP_DATA_PATH)
                # TODO: fix redudant summing
                stat["chip_stats"] = get_xarr_stats(chip_data)
                stat["chip_stats"] |= get_band_stats(chip_data, DATETIME_LABEL)

                if os.path.isdir(ANNO_DATA_PATH):
                    LOGGER.info(f"Verifying annotation data...")
                    anno_data = xr.open_zarr(ANNO_DATA_PATH)
                    # TODO: choice in weighting scheme
                    stat["anno_stats"] = get_xarr_stats(anno_data)
                    stat["anno_stats"] |= get_class_weights(anno_data)
            case _:
                raise ValueError(
                    f"Invalid file type: {SAMPLER_CONFIG['file_type']}")
        update_yaml(stat, STAT_DATA_PATH)
    except Exception as e:
        LOGGER.critical(
            f"Error in calculating sample statistics: {type(e)} {e}")
        raise e


@function_timer
def index():
    num_samples = len(gpd.read_file(META_DATA_PATH))
    samples = np.arange(num_samples)

    if SAMPLER_CONFIG["postprocess_actions"]:
        LOGGER.info("Postprocessing data in geo file...")
        # TODO: implement tif version
        chip_data = xr.open_zarr(CHIP_DATA_PATH)
        anno_data = xr.open_zarr(ANNO_DATA_PATH) if \
            os.path.exists(ANNO_DATA_PATH) else None
        sample_config = {
            "chip_data": chip_data,
            "anno_data": anno_data,
            "postprocess_actions": SAMPLER_CONFIG["postprocess_actions"],
        }
        # TODO: resample data so sample class ratios stays consistent postprocessing
        samples = postprocess_data(**sample_config)

    if SAMPLER_CONFIG["generate_time_combinations"]:
        LOGGER.info("Generating time combinations...")
        time_sample_config = {
            "samples": samples,
            "look_range": SAMPLER_CONFIG["look_range"],
            "time_step": SAMPLER_CONFIG["time_step"]
        }
        samples = generate_time_combinations(**time_sample_config)
    np.save(ALL_SAMPLE_PATH, samples)
    
    if SAMPLER_CONFIG["split_ratios"]:
        LOGGER.info(
            "Splitting sample data into training, validation, test, and predict sets...")
        train, validate, test = train_validate_test_split(samples, SAMPLER_CONFIG["split_ratios"], RANDOM_SEED)
        idx_payload = {
            TRAIN_SAMPLE_PATH: train,
            VALIDATE_SAMPLE_PATH: validate,
            TEST_SAMPLE_PATH: test
        }

        LOGGER.info("Saving sample data splits to paths...")
        for path, idx_lst in idx_payload.items():
            np.save(path, idx_lst)
        update_yaml(
            {
                "train_count": len(train),
                "validate_count": len(validate),
                "test_count": len(test)
            },
            STAT_DATA_PATH)
    else:
        LOGGER.info("Saving sample data indices to paths...")
        np.save(PREDICT_SAMPLE_PATH, samples)
        update_yaml({"predict_count": len(samples)}, STAT_DATA_PATH)
