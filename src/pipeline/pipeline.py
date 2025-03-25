import ee
import geopandas as gpd
import importlib
import multiprocessing as mp
import numpy as np
import operator
import os
import pandas as pd
import random
import shapely
import xarray as xr

from typing import Any, Optional, Literal

from config_utils import update_yaml
from constants import (
    APPEND_DIM,
    DATETIME_LABEL,
    IDX_NAME_ZFILL,
    METHOD,
    NO_DATA_VALUE,
    RANDOM_SEED,
    CLASS_LABEL,
)
from constants import (
    ALL_SAMPLE_PATH,
    ANNO_DATA_PATH,
    CHIP_DATA_PATH,
    GEO_POP_PATH,
    GEO_RAW_PATH,
    LOG_PATH,
    META_DATA_PATH,
    PREDICT_SAMPLE_PATH,
    STAT_DATA_PATH,
    TEST_SAMPLE_PATH,
    TRAIN_SAMPLE_PATH,
    VALIDATE_SAMPLE_PATH,
)
from pipeline.logging import function_timer, get_logger
from pipeline.downloader import Downloader
from pipeline.settings import SAMPLER_CONFIG
from pipeline.utils import (
    covering_grid,
    generate_centroid_squares,
    get_class_weights,
    get_band_stats,
    stratified_sample,
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
        projection: Optional[str] = None) -> gpd.GeoDataFrame:
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
    return geo_dataframe


@function_timer
def postprocess_data(
    chip_data: xr.DataArray,
    anno_data: xr.DataArray | None,
    postprocess_actions: list[str]):
    stat_data = {}

    for action in postprocess_actions:
        match action:
            case "band_mean_stdv":
                match SAMPLER_CONFIG["file_type"]:
                    case "ZARR":
                        LOGGER.info(f"Verifying chip data...")
                        # TODO: fix redudant summing
                        stat_data["chip_stats"] = {"count": len(chip_data)}
                        stat_data["chip_stats"] |= get_band_stats(chip_data)
                    case _:
                        raise ValueError(
                            f"Invalid file type: {SAMPLER_CONFIG['file_type']}")
            case "class_counts":
                gdf = gpd.read_file(GEO_POP_PATH)
                gdf.loc[:, "area"] = gdf.area
                
                groupby = gdf.loc[:, [CLASS_LABEL, "area"]].groupby(CLASS_LABEL)
                stat_data["class_geo_count"] = groupby.size().to_dict()
                stat_data["class_geo_area"] = groupby.sum()["area"].to_dict()
                if os.path.isdir(ANNO_DATA_PATH):
                    LOGGER.info(f"Verifying annotation data...")
                    # TODO: choice in weighting scheme
                    stat_data["anno_stats"] = get_class_weights(anno_data)
            case _:
                raise ValueError(f"Invalid action: {action}")
    update_yaml(stat_data, STAT_DATA_PATH)


@function_timer
def generate_squares(
        geo_dataframe: gpd.GeoDataFrame,
        method: Literal["convering_grid", "random", "centroid"],
        meter_edge_size: int,
        squares_config: dict) -> gpd.GeoDataFrame:
    LOGGER.info(f"Generating squares from sample points via {method}...")
    match method:
        case "covering_grid":
            gdf = covering_grid(geo_dataframe,
                                meter_edge_size,
                                overlap=squares_config.get("overlap", 0),
                                year_offset=squares_config.get("year_offset", 0))
        case "random":
            raise NotImplementedError
        case "centroid":
            gdf = generate_centroid_squares(
                geo_dataframe,
                meter_edge_size
            )
        case _:
            raise ValueError(f"Invalid method: {method}")

    return gdf


@function_timer
def generate_annotation_data(
        annotator: str,
        annotator_kwargs: dict,
        population_gdf: gpd.GeoDataFrame,
        squares_gdf: gpd.GeoDataFrame,
        pixel_edge_size: int,
        anno_data_path: str,
        num_workers: int,
        io_limit: int):
    num_samples = len(squares_gdf)
    class_names = sorted(population_gdf[CLASS_LABEL].unique())
    annotator = getattr(importlib.import_module("pipeline.annotators"), annotator)

    manager = mp.Manager()
    index_queue = manager.Queue()
    io_lock = manager.Lock()

    LOGGER.info(f"Starting parallel process for {num_samples} samples using {num_workers}...")
    [index_queue.put(i) for i in range(num_samples)]
    [index_queue.put(None) for _ in range(num_workers)]
    annotators = set()
    for _ in range(num_workers):
        p = mp.Process(
            target=annotator,
            kwargs={
                "population_gdf": population_gdf,
                "squares_gdf": squares_gdf,
                "class_names": class_names,
                "pixel_edge_size": pixel_edge_size,
                "anno_data_path": anno_data_path,
                "io_limit": io_limit,
                "io_lock": io_lock,
                "index_queue": index_queue} | annotator_kwargs,
            daemon=True)
        p.start()
        annotators.add(p)
    [r.join() for r in annotators]


def generate_image_chip_data(downloader_kwargs: dict):
    downloader = Downloader(**downloader_kwargs)
    downloader.start()


@function_timer
def sample():
    try:
        LOGGER.info("Loading geo file into GeoDataFrame...")
        geo_dataframe = gpd.read_file(GEO_RAW_PATH)

        LOGGER.info("Preprocessing data in geo file...")
        if SAMPLER_CONFIG["preprocess_actions"]:
            sample_config = {
                "geo_dataframe": geo_dataframe,
                "preprocess_actions": SAMPLER_CONFIG["preprocess_actions"],
                "projection": SAMPLER_CONFIG["projection"],
            }
            geo_dataframe = preprocess_data(**sample_config)
        
        LOGGER.info(f"Saving processed geo dataframe to {GEO_POP_PATH}...")
        if SAMPLER_CONFIG["datetime_column"] is not None:
            geo_dataframe.loc[:, DATETIME_LABEL] = geo_dataframe[SAMPLER_CONFIG["datetime_column"]]
        if SAMPLER_CONFIG["class_columns"] is not None:
            geo_dataframe.loc[:, CLASS_LABEL] = geo_dataframe[SAMPLER_CONFIG["class_columns"]]\
                .apply(lambda x: '__'.join(x.astype(str)).replace(" ", "_"), axis=1)
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
        if os.path.exists(GEO_POP_PATH):
            population_gdf = gpd.read_file(GEO_POP_PATH)
        else:
            population_gdf = gpd.read_file(GEO_RAW_PATH)
        squares_gdf = gpd.read_file(META_DATA_PATH)

        LOGGER.info("Generating annotations...")
        annotation_config = {
            "annotator": SAMPLER_CONFIG["annotator"],
            "annotator_kwargs": SAMPLER_CONFIG["annotator_kwargs"],
            "population_gdf": population_gdf,
            "squares_gdf": squares_gdf,
            "pixel_edge_size": SAMPLER_CONFIG["pixel_edge_size"],
            "anno_data_path": ANNO_DATA_PATH,
            "num_workers": SAMPLER_CONFIG["num_workers"],
            "io_limit": SAMPLER_CONFIG["io_limit"],
        }
        generate_annotation_data(**annotation_config,)
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
            "buffer": SAMPLER_CONFIG["buffer"],
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
def index():
    chip_data = xr.open_dataarray(CHIP_DATA_PATH, engine='zarr')
    anno_data = xr.open_dataarray(ANNO_DATA_PATH, engine='zarr') if os.path.exists(ANNO_DATA_PATH) else None

    if SAMPLER_CONFIG["postprocess_actions"]:
        LOGGER.info("Postprocessing data in geo file...")
        # TODO: implement tif version
        sample_config = {
            "chip_data": chip_data,
            "anno_data": anno_data,
            "postprocess_actions": SAMPLER_CONFIG["postprocess_actions"],
        }
        # TODO: resample data so sample class ratios stays consistent postprocessing
        postprocess_data(**sample_config)

    num_samples = len(gpd.read_file(META_DATA_PATH))
    samples = np.arange(num_samples)
    np.save(ALL_SAMPLE_PATH, samples)
    indexer = getattr(importlib.import_module("pipeline.indexers"), SAMPLER_CONFIG["indexer"])
    
    if SAMPLER_CONFIG["split_ratios"]:
        LOGGER.info("Splitting sample data into training, validation, test, and predict sets...")
        train, validate, test = indexer(chip_data,
                                        anno_data,
                                        SAMPLER_CONFIG["split_ratios"],
                                        RANDOM_SEED,
                                        **SAMPLER_CONFIG["indexer_kwargs"])
        idx_payload = {
            TRAIN_SAMPLE_PATH: train,
            VALIDATE_SAMPLE_PATH: validate,
            TEST_SAMPLE_PATH: test
        }

        LOGGER.info("Saving sample data splits to paths...")
        for path, idx_lst in idx_payload.items():
            np.save(path, idx_lst.astype(int))
        update_yaml(
            {
                "train_count": len(train),
                "validate_count": len(validate),
                "test_count": len(test)
            },
            STAT_DATA_PATH)
    else:
        samples = indexer(chip_data,
                            anno_data,
                            SAMPLER_CONFIG["split_ratios"],
                            RANDOM_SEED,
                            **SAMPLER_CONFIG["indexer_kwargs"])

        LOGGER.info("Saving sample data indices to paths...")
        np.save(PREDICT_SAMPLE_PATH, samples)
        update_yaml({"predict_count": len(samples)}, STAT_DATA_PATH)
