import geopandas as gpd
import multiprocessing as mp
import numpy as np
import xarray as xr

from shapely.geometry import Polygon
from typing import Any, Optional, Tuple

from pipeline.logging import get_logger
from pipeline.settings import CLASS_LABEL, DATETIME_LABEL, IDX_NAME_ZFILL, LOG_PATH, METHOD, NO_DATA_VALUE
from pipeline.utils import rasterizer


LOGGER = get_logger(LOG_PATH, METHOD)


def xarr_annotator(population_gdf: gpd.GeoDataFrame,
                   square: Polygon,
                   class_names: dict[str, int],
                   pixel_edge_size: int,
                   year: int):
    # rasterizing multipolygon and clipping to square
    xarr_anno_list = []

    # class list should already be in order of index value
    for class_name in class_names:
        try:
            mask = (population_gdf[DATETIME_LABEL] == year) & \
                (population_gdf[CLASS_LABEL] == class_name)
            mp = population_gdf[mask].geometry
            if len(mp) == 0:
                annotation = np.zeros((pixel_edge_size, pixel_edge_size))
            else:
                annotation = rasterizer(
                    mp, square, pixel_edge_size, NO_DATA_VALUE, 1)
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
                          include_class_sums: bool = True):
    batch = []
    while (index := index_queue.get()) is not None:
        index_name = str(index).zfill(IDX_NAME_ZFILL)
        
        # getting annotation information from sample
        LOGGER.info(f"Rasterizing sample {index_name}...")
        target = squares_gdf.iloc[index]
        year = target[DATETIME_LABEL]
        square = target.geometry

        # class list should already be in order of index value
        LOGGER.info(f"Creating annotations for sample {index_name} from class list...")
        xarr_anno = xarr_annotator(population_gdf,
                                   square,
                                   class_names,
                                   pixel_edge_size,
                                   year)

        xarr_anno.name = index_name
        if include_class_sums:
            xarr_anno.attrs["class_sums"] = xarr_anno.sum(dim=["y", "x"]).values

        # writing in batches to avoid io bottleneck
        LOGGER.info(f"Appending rasterized sample {index_name} of shape {xarr_anno.shape} to batch...")
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


def multi_year_xarr_annotator(population_gdf: gpd.GeoDataFrame,
                              squares_gdf: gpd.GeoDataFrame,
                              class_names: dict[str, int],
                              pixel_edge_size: int,
                              anno_data_path: str,
                              io_limit: int,
                              io_lock: Any,
                              index_queue: Optional[mp.Queue],
                              year_range: Tuple,
                              include_class_sums: bool = True):
    years = range(year_range[0], year_range[1])
    
    batch = []
    while (index := index_queue.get()) is not None:
        year_batch = []
        for year in years:
            index_name = str(index).zfill(IDX_NAME_ZFILL)
            
            # getting annotation information from sample
            LOGGER.info(f"Rasterizing sample {index_name}...")
            target = squares_gdf.iloc[index]
            square = target.geometry

            # rasterizing multipolygon and clipping to square
            LOGGER.info(f"Creating annotations for sample {index_name} from class list...")
            xarr_anno = xarr_annotator(population_gdf,
                                       square,
                                       class_names,
                                       pixel_edge_size,
                                       year)
            year_batch.append(xarr_anno)
        xarr_years = xr.concat(year_batch, dim=DATETIME_LABEL)

        xarr_years.name = index_name
        if include_class_sums:
            xarr_years.attrs["class_sums"] = xarr_years.sum(dim=["y", "x"]).values

        # writing in batches to avoid io bottleneck
        LOGGER.info(f"Appending rasterized sample {index_name} of shape {xarr_years.shape} to batch...")
        batch.append(xarr_years)
        if len(batch) == io_limit:
            xarr_years = xr.merge(batch)
            with io_lock:
                xarr_years.to_zarr(store=anno_data_path, mode="a")
            batch.clear()
            indices = list(xarr_years.data_vars)
            LOGGER.info(f"Rasterized sample batch submitted... {indices}")

    # writing remaining batch
    if len(batch) > 0:
        with io_lock:
            xarr_years = xr.merge(batch)
            xarr_years.to_zarr(store=anno_data_path, mode="a")
