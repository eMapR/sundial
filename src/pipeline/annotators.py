import geopandas as gpd
import multiprocessing as mp
import numpy as np
import os
import xarray as xr

from shapely.geometry import Polygon
from typing import Any, Optional, Tuple

from constants import APPEND_DIM, CLASS_LABEL, DATETIME_LABEL, IDX_NAME_ZFILL, LOG_PATH, METHOD, NO_DATA_VALUE
from pipeline.logging import get_logger
from pipeline.utils import rasterizer


LOGGER = get_logger(LOG_PATH, METHOD)


def xarr_annotator(population_gdf: gpd.GeoDataFrame,
                   square: Polygon,
                   class_names: dict[str, int],
                   pixel_edge_size: int,
                   datetime: int | None):
    # rasterizing multipolygon and clipping to square
    xarr_anno_list = []

    # class list should already be in order of index value
    for class_name in class_names:
        try:
            mask = population_gdf[CLASS_LABEL] == class_name
            if datetime is not None:
                mask &= population_gdf[DATETIME_LABEL] == datetime
            mp = population_gdf[mask].geometry
            if len(mp) == 0:
                annotation = np.zeros((pixel_edge_size, pixel_edge_size))
            else:
                annotation = rasterizer(mp, square, pixel_edge_size, NO_DATA_VALUE, 1)
        except Exception as e:
            raise e
        # TODO: add support for tif
        annotation = xr.DataArray(annotation, dims=["y", "x"])

        xarr_anno_list.append(annotation)
    xarr_anno = xr.concat(xarr_anno_list, dim=CLASS_LABEL)
    return xarr_anno


def single_xarr_annotator(population_gdf: gpd.GeoDataFrame,
                          squares_gdf: gpd.GeoDataFrame,
                          class_names: dict[str, int],
                          pixel_edge_size: int,
                          anno_data_path: str,
                          io_limit: int,
                          io_lock: Any,
                          index_queue: Optional[mp.Queue],
                          **kwargs):
    batch = []
    batch_names = []
    while (index := index_queue.get()) is not None:
        batch_names.append(index)
        # getting annotation information from sample
        
        LOGGER.info(f"Rasterizing sample {index:08d}...")
        target = squares_gdf.iloc[index]
        if DATETIME_LABEL in squares_gdf.columns:
            datetime = target[DATETIME_LABEL]
        else:
            datetime = None
        square = target.geometry

        # class list should already be in order of index value
        LOGGER.info(f"Creating annotations for sample {index:08d} from class list...")
        xarr_anno = xarr_annotator(population_gdf,
                                   square,
                                   class_names,
                                   pixel_edge_size,
                                   datetime)
        xarr_anno = xarr_anno.assign_coords({APPEND_DIM: index})

        # writing in batches to avoid io bottleneck
        LOGGER.info(f"Appending rasterized sample {index:08d} of shape {xarr_anno.shape} to batch...")
        batch.append(xarr_anno)
        if len(batch) == io_limit:
            xarr_anno = xr.concat(batch, dim=APPEND_DIM)
            with io_lock:
                if os.path.exists(anno_data_path):
                    xarr_anno.to_zarr(store=anno_data_path, append_dim=APPEND_DIM, mode="a")
                else:
                    xarr_anno.to_zarr(store=anno_data_path)
            LOGGER.info(f"Rasterized sample batch submitted... {batch_names}")
            
            batch.clear()
            batch_names.clear()

    # writing remaining batch
    if len(batch) > 0:
        xarr_anno = xr.concat(batch, dim=APPEND_DIM)
        with io_lock:
            if os.path.exists(anno_data_path):
                xarr_anno.to_zarr(store=anno_data_path, append_dim=APPEND_DIM, mode="a")
            else:
                xarr_anno.to_zarr(store=anno_data_path)


def multi_year_xarr_annotator(population_gdf: gpd.GeoDataFrame,
                              squares_gdf: gpd.GeoDataFrame,
                              class_names: dict[str, int],
                              pixel_edge_size: int,
                              anno_data_path: str,
                              io_limit: int,
                              io_lock: Any,
                              index_queue: Optional[mp.Queue],
                              year_range: Tuple,
                              **kwargs):
    years = range(year_range[0], year_range[1])
    
    batch = []
    batch_names = []

    while (index := index_queue.get()) is not None:
        LOGGER.info(f"Rasterizing sample {index:08d}...")
        batch_names.append(index)
        year_batch = []
        for year in years:
            # getting annotation information from sample
            target = squares_gdf.iloc[index]
            square = target.geometry

            # rasterizing multipolygon and clipping to square
            xarr_anno = xarr_annotator(population_gdf,
                                       square,
                                       class_names,
                                       pixel_edge_size,
                                       year)
            year_batch.append(xarr_anno)
        xarr_years = xr.concat(year_batch, dim=DATETIME_LABEL).assign_coords({APPEND_DIM: index})

        # writing in batches to avoid io bottleneck
        LOGGER.info(f"Appending rasterized sample {index:08d} of shape {xarr_years.shape} to batch...")
        batch.append(xarr_years)

        if len(batch) == io_limit:
            xarr_years = xr.concat(batch, dim=APPEND_DIM, coords='all')
            with io_lock:
                if os.path.exists(anno_data_path):
                    xarr_years.to_zarr(store=anno_data_path, append_dim=APPEND_DIM, mode="a")
                else:
                    xarr_years.to_zarr(store=anno_data_path)
            LOGGER.info(f"Rasterized sample batch submitted... {batch_names}")
            batch.clear()
            batch_names.clear()

    # writing remaining batch
    if len(batch) > 0:
        with io_lock:
            xarr_years = xr.concat(batch, dim=APPEND_DIM, )
            if os.path.exists(anno_data_path):
                xarr_years.to_zarr(store=anno_data_path, append_dim=APPEND_DIM, mode="a")
            else:
                xarr_years.to_zarr(store=anno_data_path)