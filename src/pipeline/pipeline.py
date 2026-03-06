import geopandas as gpd
import os
import xarray as xr

from typing import Any, Optional, Literal

from config_utils import update_yaml
from constants import (
    ANNOTATIONS_PATH,
    IMAGERY_PATH,
    GEO_PROC_PATH,
    LOG_PATH,
    METHOD,
    STAT_DATA_PATH,
)
from pipeline.logging import function_timer, get_logger
from pipeline.downloader import Downloader
from pipeline.settings import PIPELINE_CONFIG
from pipeline.utils import (
    get_band_stats,
)
from config_utils import dynamic_import


LOGGER = get_logger(LOG_PATH, METHOD)


@function_timer
def generate_stats(
    imagery_da: xr.DataArray,
    annotations_da: xr.DataArray | None,
    stat_actions: list[str],
    label_column: str):
    stat_data = {}

    for action in postprocess_actions:
        match action:
            case "band_mean_stdv":
                match PIPELINE_CONFIG["file_type"]:
                    case "ZARR":
                        LOGGER.info(f"Verifying chip data...")
                        # TODO: fix redudant summing
                        stat_data["chip_stats"] |= get_band_stats(imagery_da)
                    case _:
                        raise ValueError(
                            f"Invalid file type: {PIPELINE_CONFIG['file_type']}")
            case "class_counts":
                gdf = gpd.read_file(GEO_PROC_PATH)
                gdf.loc[:, "area"] = gdf.area
                
                groupby = gdf.loc[:, [label_column, "area"]].groupby(label_column)
                stat_data["class_geo_count"] = groupby.size().to_dict()
                stat_data["class_geo_area"] = groupby.sum()["area"].to_dict()
            case _:
                raise ValueError(f"Invalid action: {action}")
    update_yaml(stat_data, STAT_DATA_PATH)


@function_timer
def download():
    try:
        LOGGER.info("Generating image chips...")
        downloader_kwargs = {
            **PIPELINE_CONFIG,
            "geo_proc_data": gpd.read_file(GEO_PROC_PATH),
            "imagery_path": IMAGERY_PATH,
            "logger": LOGGER,
        }
        downloader = Downloader(**downloader_kwargs)
        downloader.start()
    except Exception as e:
        LOGGER.critical(f"Failed to download images: {type(e)} {e}")
        raise e

    
@function_timer
def annotate():
    try:
        LOGGER.info("Generating annotations...")
        annotation_config = {
            **PIPELINE_CONFIG,
            "geo_proc_data": gpd.read_file(GEO_PROC_PATH),
            "annotations_path": ANNOTATIONS_PATH,
            "logger": LOGGER,
        }
        annotator = dynamic_import(annotation_config.pop("annotator"), annotation_config)
        annotator.start()
    except Exception as e:
        LOGGER.critical(f"Failed to generate annotations: {type(e)} {e}")
        raise e


@function_timer
def stats():
    imagery_da = xr.open_dataarray(IMAGERY_PATH, engine='zarr')
    annotations_da = xr.open_dataarray(ANNOTATIONS_PATH, engine='zarr') if os.path.exists(ANNOTATIONS_PATH) else None

    LOGGER.info("Postprocessing data in geo file...")
    stat_config = {
        "imagery_da": imagery_da,
        "annotations_da": annotations_da,
        **PIPELINE_CONFIG,
    }
    generate_stats(**stat_config)
