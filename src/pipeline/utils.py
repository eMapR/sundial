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

from dataclasses import dataclass, field
from datetime import datetime
from numpy.lib import recfunctions as rfn
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely import STRtree
from shapely.geometry import box
from typing import Literal, Optional, Tuple

from config_utils import update_yaml


class ParallelGridAlign:
    def __init__(self, **kwargs):
        self._geo_proc_data = kwargs.get("geo_proc_data")
        self._epsg_str = self._geo_proc_data.crs.to_string()
        
        self._dtype = np.dtype(kwargs.get("dtype"))
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
        dims = ["band", "time", "lat", "lon"]
        dummy_zarr(self._dtype, self._chunk_sizes, shape, dims, self._lat_coords, self._lon_coords, self._source_path)

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
            if array.dtype.names is not None:
                names = array.dtype.names
                H, W = array.shape

                array = rfn.structured_to_unstructured(array)
                array = array.reshape(H, W, self._chunk_sizes[1], self._chunk_sizes[0])
                array = array.transpose(3, 2, 0, 1)
            array = array.astype(self._dtype)

            # NOTE: all arange calls may have errors at large scales consider linspace but for now this is fine for equal area proj.
            lats = np.arange(translateY, translateY - self._grid_y_size, -self._scale)
            lons = np.arange(translateX, translateX + self._grid_x_size, self._scale)

            chunk_da = xr.DataArray(
                array,
                dims=["band", "time", "lat", "lon"],
                coords={
                    "band": list(range(self._chunk_sizes[0])),
                    "time": list(range(self._chunk_sizes[1])),
                    "lat": lats,
                    "lon": lons
                },
                name="dat"
            )
            lat_start = np.searchsorted(-self._lat_coords, -lats[0])
            lon_start = np.searchsorted(self._lon_coords, lons[0])

            region = {
                "band": slice(0, self._chunk_sizes[0]),
                "time": slice(0, self._chunk_sizes[1]),
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


@dataclass
class BandStats:
    n:     np.ndarray = field(default_factory=lambda: np.zeros(6))
    mean:  np.ndarray = field(default_factory=lambda: np.zeros(6))
    M2:    np.ndarray = field(default_factory=lambda: np.zeros(6))

    def update(self, subset: xr.DataArray):
        """Merge a new chunk's stats into the accumulator."""
        b_mean = subset.mean(dim=["time", "lat", "lon"], skipna=True).values
        b_var = subset.var( dim=["time", "lat", "lon"], skipna=True).values
        b_n = subset.count(dim=["time", "lat", "lon"]).values

        delta = b_mean - self.mean
        combined_n = self.n + b_n

        self.mean = (self.n * self.mean + b_n * b_mean) / np.where(combined_n > 0, combined_n, 1)
        self.M2 += b_var * b_n + delta**2 * self.n * b_n / np.where(combined_n > 0, combined_n, 1)
        self.n = combined_n

    @property
    def std(self):
        return np.sqrt(self.M2 / np.where(self.n > 1, self.n - 1, 1))


def filter_chunks(chunks, geo_proc_data, grid_y_size, grid_x_size):
    res = []
    _spatial_index = STRtree(geo_proc_data.geometry)
    for c in chunks:
        ty, tx = c
        b = box(tx, ty - grid_y_size, tx + grid_x_size, ty)
        hits = _spatial_index.query(b, predicate="intersects")
        if hits.size > 0:
            res.append(c)
    return res


def chunk_bounds(total_bounds, grid_y_size, grid_x_size):
    lats = np.arange(total_bounds[3], total_bounds[1], -grid_y_size)
    lons = np.arange(total_bounds[0], total_bounds[2], grid_x_size)
    chunks = np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2)
    return chunks.tolist()


def coord_bounds(total_bounds, grid_y_size, grid_x_size, scale):
    grid_y_size, grid_x_size
    
    ybuf = grid_y_size - ((total_bounds[3] - total_bounds[1]) % grid_y_size)
    xbuf = grid_x_size - ((total_bounds[2] - total_bounds[0]) % grid_x_size)
    
    lat_coords = np.arange(total_bounds[3], total_bounds[1] - ybuf, -scale)
    lon_coords = np.arange(total_bounds[0], total_bounds[2] + xbuf, scale)
    return lat_coords, lon_coords


def dummy_zarr(dtype, chunk_sizes, shape, dims, y_coords, x_coords, out_path):
    dummy_da = xr.DataArray(
        da.full(
            shape,
            np.nan,
            dtype=dtype,
            chunks=chunk_sizes,
        ),
        dims=dims,
        coords={"band": list(range(chunk_sizes[0])), "time": list(range(chunk_sizes[1])), "lat": y_coords, "lon": x_coords},
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


def get_band_stats(data: xr.DataArray, geo_proc_data: gpd.GeoDataFrame, filter_intersect: bool, chunk_sizes: tuple, scale, **kwargs) -> tuple[list[float], list[float]]:
    if filter_intersect:
        total_bounds = [
            data.coords["lon"][0],
            data.coords["lat"][-1],
            data.coords["lon"][-1],
            data.coords["lat"][0]
        ]
        grid_y_size = chunk_sizes[-2]*scale
        grid_x_size = chunk_sizes[-1]*scale
        
        chunks = chunk_bounds(total_bounds, grid_y_size, grid_x_size)
        chunks = filter_chunks(chunks, geo_proc_data, grid_y_size, grid_x_size)
        
        stats = BandStats()
        for ty, tx in chunks:
            subset = data.sel(lat=slice(ty, ty - grid_y_size), lon=slice(tx, tx + grid_x_size))
            stats.update(subset.compute())

        return {"band_means": stats.mean.tolist(), "band_stds": stats.std.tolist()}

    else:     
        means = data.mean(dim=["time", "lat", "lon"], skipna=True)
        stds  = data.std( dim=["time", "lat", "lon"], skipna=True)
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


def generate_stats(
    imagery_da: xr.DataArray,
    annotations_da: xr.DataArray | None,
    geo_proc_data: gpd.GeoDataFrame,
    stat_data_path: str,
    **kwargs):
    stat_data = {}
    
    for action in kwargs.get("stats_actions", []):
        match action:
            case "band_mean_stdv":
                stat_data["chip_stats"] = get_band_stats(imagery_da, geo_proc_data, **kwargs)
            case "class_counts":
                label_column = kwargs.get("annotator")["init_args"]["label_column"]
                geo_proc_data.loc[:, "area"] = geo_proc_data.area
                
                groupby = geo_proc_data.loc[:, [label_column, "area"]].groupby(label_column)
                stat_data["class_geo_count"] = groupby.size().to_dict()
                stat_data["class_geo_area"] = groupby.sum()["area"].to_dict()
            case _:
                raise ValueError(f"Invalid action: {action}")
    update_yaml(stat_data, stat_data_path)
