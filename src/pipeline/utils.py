import dask.array as da
import geopandas as gpd
import ee
import pandas as pd
import multiprocessing as mp
import numpy as np
import os
import shapely
import time
import utm
import xarray as xr

from datetime import datetime
from numpy.lib import recfunctions as rfn
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely import STRtree
from shapely.geometry import box
from typing import Literal, Optional, Tuple


class ParallelGridAlign:
    def __init__(self, **kwargs):
        self._geo_proc_data = kwargs.get("geo_proc_data")
        self._epsg_str = self._geo_proc_data.crs.to_string()
        
        self._chunk_sizes = kwargs.get("chunk_sizes")
        self._scale = kwargs.get("scale")
        
        self._num_workers = kwargs.get("num_workers")
        self._io_limit = kwargs.get("io_limit")
        self._logger = kwargs.get("logger")
    
    def start(self):
        self._align()
        self._watcher()
    
    def _align(self) -> None:
        self._total_bounds = self._geo_proc_data.total_bounds
        self._grid_y_size = self._chunk_sizes[-2] * self._scale
        self._grid_x_size = self._chunk_sizes[-1] * self._scale
        
        self._chunks = chunk_bounds(self._total_bounds, self._grid_y_size, self._grid_x_size)
        if self._filter_intersect:
            self._chunks = filter_chunks(self._chunks, self._geo_proc_data, self._grid_y_size, self._grid_x_size)
        self._lat_coords, self._lon_coords = coord_bounds(self._total_bounds, self._grid_y_size, self._grid_x_size, self._scale)

        shape = (*self._chunk_sizes[:2], len(self._lat_coords), len(self._lon_coords))
        dims = ["time", "band", "lat", "lon"]
        dummy_zarr(self._chunk_sizes, shape, dims, self._lat_coords, self._lon_coords, self._source_path)

    def _watcher(self) -> None:
        with mp.Manager() as manager:
            self._chunk_queue = manager.Queue()
            self._result_queue = manager.Queue()
            self._report_queue = manager.Queue()

            reporter = mp.Process(
                target=self._reporter,
                args=[],
                daemon=True)
            reporter.start()
            
            for chunk in self._chunks:
                self._chunk_queue.put(chunk)               

            for _ in range(self._num_workers):
                self._chunk_queue.put(None)

            consumers = set()
            self._report_queue.put(("INFO", f"Starting {len(self._chunks)} chunk tasks..."))
            for consumer_index in range(self._num_workers):
                consumer = mp.Process(
                    target=self._consumer,
                    kwargs={"consumer_index": consumer_index},
                    daemon=True)
                consumer.start()
                consumers.add(consumer)

            tasks_completed = 0
            consumers_completed = 0
            while consumers_completed < self._num_workers:
                result = self._result_queue.get()
                if result is not None:
                    tasks_completed += 1
                    self._report_queue.put(("INFO", f"{tasks_completed}/{len(self._chunks)} tasks completed. {result}"))
                else:
                    consumers_completed += 1
                    self._report_queue.put(("INFO", f"{consumers_completed}/{self._num_workers} consumers completed."))

            self._report_queue.put(None)
            for c in consumers:
                c.join()
            reporter.join()
        
    def _write_array_batch(self, chunk_batch) -> None:
        for array, translateY, translateX  in chunk_batch:
            arr = rfn.structured_to_unstructured(array).transpose(2, 0, 1)
            arr = arr.reshape(self._chunk_sizes)

            lats = np.arange(translateY, translateY - self._grid_y_size, -self._scale)
            lons = np.arange(translateX, translateX + self._grid_x_size, self._scale)

            chunk_da = xr.DataArray(
                arr,
                dims=["time", "band", "lat", "lon"],
                coords={
                    "time": list(range(self._chunk_sizes[0])),
                    "band": list(range(self._chunk_sizes[1])),
                    "lat": lats,
                    "lon": lons
                },
                name="dat"
            )
            lat_start = np.searchsorted(-self._lat_coords, -lats[0])
            lon_start = np.searchsorted(self._lon_coords, lons[0])

            region = {
                "time": slice(0, self._chunk_sizes[0]),
                "band": slice(0, self._chunk_sizes[1]),
                "lat": slice(lat_start, lat_start + len(lats)),
                "lon": slice(lon_start, lon_start + len(lons)),
            }

            chunk_da.to_zarr(self._source_path, region=region)
            self._result_queue.put((translateY, translateX))

    def _reporter(self) -> None:
        while (report := self._report_queue.get()) is not None:
            level, message = report
            if self._logger is not None:
                match level:
                    case "DEBUG":
                        self._logger.debug(message)
                    case "INFO":
                        self._logger.info(message)
                    case "WARNING":
                        self._logger.warning(message)
                    case "ERROR":
                        self._logger.error(message)
                    case "CRITICAL":
                        self._logger.critical(message)
            else:
                print(level, message)
        self._logger.info(("INFO", f"Reporter completed. Ending..."))


def filter_chunks(chunks, geo_proc_data, grid_y_size, grid_x_size):
    res = []
    _spatial_index = STRtree(geo_proc_data.geometry)
    for c in chunks:
        ty, tx = c
        b = box(tx, ty - grid_y_size, tx + grid_x_size, ty)
        hits = _spatial_index.query(b, predicate="intersects")
        if hits.size > 0:
            res.append(c)
    return np.array(res)


def chunk_bounds(total_bounds, grid_y_size, grid_x_size):
    lats = np.arange(total_bounds[3], total_bounds[1], -grid_y_size)
    lons = np.arange(total_bounds[0], total_bounds[2], grid_x_size)
    chunks = np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2)
    return chunks


def coord_bounds(total_bounds, grid_y_size, grid_x_size, scale):
    grid_y_size, grid_x_size
    
    ybuf = grid_y_size - ((total_bounds[3] - total_bounds[1]) % grid_y_size)
    xbuf = grid_x_size - ((total_bounds[2] - total_bounds[0]) % grid_x_size)
    
    lat_coords = np.arange(total_bounds[3], total_bounds[1] - ybuf, -scale)
    lon_coords = np.arange(total_bounds[0], total_bounds[2] + xbuf, scale)
    return lat_coords, lon_coords


def dummy_zarr(chunk_sizes, shape, dims, y_coords, x_coords, out_path):
    dummy_da = xr.DataArray(
        da.full(
            shape,
            np.nan,
            dtype=np.float32,
            chunks=chunk_sizes,
        ),
        dims=dims,
        coords={"time": list(range(chunk_sizes[0])), "band": list(range(chunk_sizes[1])), "lat": y_coords, "lon": x_coords},
        name="dat",
    )

    dummy_da.to_zarr(out_path, compute=False, mode="w")


def clip_xy_xarray(xarr: xr.DataArray, 
                   pixel_edge_size: int) -> xr.DataArray:
    x_diff = xarr["lon"].size - pixel_edge_size
    y_diff = xarr["lat"].size - pixel_edge_size
    
    assert x_diff > 0 and y_diff > 0, "image must be larger than clip size"

    x_start = x_diff // 2 
    x_end = xarr["lon"].size - (x_diff - x_start)

    y_start = y_diff // 2
    y_end = xarr["lat"].size - (y_diff - y_start)

    return xarr.sel(x=slice(x_start, x_end), y=slice(y_start, y_end))


def pad_xy_xarray(
        xarr: xr.DataArray,
        pixel_edge_size: int,
        no_data_value: float | int) -> xr.DataArray:
    x_diff = pixel_edge_size - xarr["lon"].size
    y_diff = pixel_edge_size - xarr["lat"].size

    assert x_diff > 0 and y_diff > 0, "image must be smaller than pad size"

    x_start = x_diff // 2
    x_end = x_diff - x_start

    y_start = y_diff // 2
    y_end = y_diff - y_start

    xarr = xarr.pad(
        x=(x_start, x_end),
        y=(y_start, y_end),
        keep_attrs=True,
        mode="constant",
        constant_values=no_data_value)
    return xarr


def get_utm_zone(point_coords: list[tuple[float]]) -> int:
    revserse_point = reversed(point_coords[0])
    utm_zone = utm.from_latlon(*revserse_point)[-2:]
    epsg_prefix = "EPSG:326" if point_coords[1] > 0 else "EPSG:327"
    epsg_code = f"{epsg_prefix}{utm_zone[0]}"

    return epsg_code


def get_class_weights(data: xr.DataArray) -> tuple[list[float], list[float]]:
    dims=["time", "lat", "lon"]
    sums = data.sum(dim=dims)
    class_totals = np.array(sums.values)
    class_probs = class_totals / class_totals.sum()
    if class_probs.sum() > 0:
        weights = 1 / class_probs
    else:
        weights = np.repeat(1, len(class_probs))
    return {"totals": class_totals.tolist(), "weights": weights.tolist()}


def get_band_stats(data: xr.DataArray) -> tuple[list[float], list[float]]:
    means = data.mean(dim=["time", "lat", "lon"])
    stds = data.std(dim=["time", "lat", "lon"])
    return {"band_means": means.values.tolist(), "band_stds": stds.values.tolist()}


def rasterizer(polygons: gpd.GeoSeries,
               bounds: list[float],
               chunk_y_size: int,
               chunk_x_size: int,
               fill: int | float,
               default_value: int | float):

    transform = from_bounds(*bounds, chunk_y_size, chunk_x_size)
    raster = rasterize(
        shapes=polygons,
        out_shape=(chunk_y_size, chunk_x_size),
        fill=fill,
        transform=transform,
        default_value=default_value,
        dtype="float32")

    return raster
