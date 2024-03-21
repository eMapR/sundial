import ee
import pandas as pd
import numpy as np
import time
import utm
import xarray as xr

from datetime import datetime
from ltgee import LandTrendr
from typing import Optional

from .settings import MASK_LABELS, NO_DATA_VALUE


def lt_image_generator(
        square_coords: list[tuple[float, float]],
        start_date: datetime,
        end_date: datetime,
        scale: int,
        projection: str,
        mask_labels: list[str] = MASK_LABELS) -> ee.Image:
    if projection is None or projection != "EPSG:4326":
        even_odd = False
    else:
        even_odd = True
    square = ee.Geometry.Polygon(
        square_coords, proj=projection, evenOdd=even_odd)
    lt = LandTrendr(
        start_date=start_date,
        end_date=end_date,
        area_of_interest=square,
        mask_labels=mask_labels,
        run=False
    )
    collection = lt.build_sr_collection()
    size = collection.size().getInfo()

    old_band_names = [f"{str(i)}_{band}" for i in range(size)
                      for band in lt._band_names]
    new_band_names = [f"{str(start_date.year + i)}_{band}" for i in range(size)
                      for band in lt._band_names]

    image = collection\
        .toBands()\
        .select(old_band_names, new_band_names)\
        .reproject(crs=projection, scale=scale)\
        .clipToBoundsAndScale(geometry=square, scale=scale)

    return image


def zarr_reshape(
        arr: np.ndarray,
        index: str,
        pixel_edge_size: int,
        square_name: str,
        point_name: str,
        attributes: Optional[dict] = None) -> xr.DataArray:

    # unflattening the array to shape (year, x, y, band)
    years, bands = zip(*[b.split('_')
                       for b in arr.dtype.names if b != "overlap"])
    years = set(years)
    bands = set(bands)
    xr_list = [
        xr.DataArray(
            np.dstack([arr[f"{y}_{b}"] for b in bands]),
            dims=['y', 'x', "band"]
        ).astype(float)
        for y in years]
    xarr = xr.concat(xr_list, dim="year")

    # adding strata data as attributes
    xarr.name = str(index)
    new_attrs = attributes | {"point": point_name, "square": square_name}
    xarr.attrs.update(**new_attrs)

    # padding the xarray to the edge size to maintain consistent image size in zarr
    if pixel_edge_size > min(xarr["x"].size,  xarr["y"].size):
        xarr = pad_xy_xarray(xarr, pixel_edge_size)
    if pixel_edge_size < max(xarr["x"].size,  xarr["y"].size):
        xarr = clip_xy_xarray(xarr, pixel_edge_size)

    return xarr.chunk(chunks={"year": 1})


def clip_xy_xarray(xarr: xr.DataArray,
                   pixel_edge_size: int) -> xr.DataArray:
    x_diff = xarr["x"].size - pixel_edge_size
    y_diff = xarr["y"].size - pixel_edge_size

    x_start = x_diff // 2 if x_diff > 0 else 0
    x_end = x_diff - x_start if x_diff > 0 else 0

    y_start = y_diff // 2 if y_diff > 0 else 0
    y_end = y_diff - y_start if y_diff > 0 else 0

    return xarr.sel(x=slice(x_start, xarr["x"].size-x_end), y=slice(y_start, xarr["y"].size-y_end))


def pad_xy_xarray(
        xarr: xr.DataArray,
        pixel_edge_size: int) -> xr.DataArray:
    x_diff = pixel_edge_size - xarr["x"].size
    y_diff = pixel_edge_size - xarr["y"].size

    x_start = x_diff // 2 if x_diff > 0 else 0
    x_end = x_diff - x_start if x_diff > 0 else 0

    y_start = y_diff // 2 if y_diff > 0 else 0
    y_end = y_diff - y_start if y_diff > 0 else 0

    xarr = xarr.pad(
        x=(x_start, x_end),
        y=(y_start, y_end),
        keep_attrs=True,
        mode="constant",
        constant_values=NO_DATA_VALUE)
    return xarr


def generate_coords_name(coords: tuple[float], index) -> str:
    if len(coords) > 2:
        coords = coords[:-1]
    return f"{index}-" + "_".join([f"x{x}y{y}" for x, y in coords])


def get_utm_zone(point_coords: list[tuple[float]]) -> int:
    revserse_point = reversed(point_coords[0])
    utm_zone = utm.from_latlon(*revserse_point)[-2:]
    epsg_prefix = "EPSG:326" if point_coords[1] > 0 else "EPSG:327"
    epsg_code = f"{epsg_prefix}{utm_zone[0]}"

    return epsg_code


def parse_meta_data(
        meta_data: pd.DataFrame,
        index: int,
        look_years: int,
        start_month: int,
        start_day: int,
        end_month: int,
        end_day: int) -> tuple[list[tuple[float, float]],
                               tuple[float, float],
                               str,
                               list[tuple[float, float]],
                               str,
                               datetime | None,
                               datetime | None,
                               dict]:
    square = meta_data.iloc[index].loc["geometry"]
    square_coords = list(square.boundary.coords)
    point_coords = list(square.centroid.coords)

    # generating start and end date from year attribute and back step
    end_year = meta_data.iloc[index].loc["year"]

    end_date = datetime(end_year, end_month, end_day)
    start_date = datetime(end_year - look_years, start_month, start_day)

    data_vars = meta_data.iloc[index].to_dict()
    attributes = {k: v for k, v in data_vars.items()
                  if all([s not in k for s in ["geometry", "square", "point"]])}

    return square_coords, \
        point_coords, \
        start_date, \
        end_date, \
        attributes


def function_timer(logger=None):
    def wrapper(func):
        def timer(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if logger:
                logger.info(
                    f"{func.__name__} took {(end - start)/60:.3f} minutes to complete.")
            else:
                print(
                    f"{func.__name__} took {(end - start)/60:.3f} minutes to complete.")
            return result
        return timer
    return wrapper

@function_timer()
def get_mean_std(chip_data_path: str):
    data = xr.open_zarr(chip_data_path)
    data = data.to_dataarray(dim="chip")
    means = data.mean(dim=["chip", "year", "y", "x"])
    stds = data.std(dim=["chip", "year", "y", "x"])
    return means, stds
