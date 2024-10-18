import ee
import pandas as pd
import numpy as np
import time
import utm
import xarray as xr

from datetime import datetime
from ltgee import LandsatComposite
from typing import Optional, Generator, Tuple

from pipeline.logger import get_logger
from pipeline.settings import NO_DATA_VALUE, LOG_PATH, METHOD, DATETIME_LABEL


def lt_medoid_image_generator(
        square_coords: list[tuple[float, float]],
        start_date: datetime,
        end_date: datetime,
        scale: int,
        projection: str,
        mask_labels: list[str] = ["snow", "cloud", "shadow"]) -> ee.Image:
    # TODO: actually parse the projection string
    even_odd = (projection == "EPSG:4326")
    square = ee.Geometry.Polygon(
        square_coords, proj=projection, evenOdd=even_odd)
    collection = LandsatComposite(
        start_date=start_date,
        end_date=end_date,
        area_of_interest=square,
        mask_labels=mask_labels,
    )
    size = 1 + end_date.year - start_date.year

    old_band_names = [f"{str(i)}_{band}" for i in range(size)
                      for band in collection._band_names]
    new_band_names = [f"{str(start_date.year + i)}_{band}" for i in range(size)
                      for band in collection._band_names]

    # TODO: fix hacky filter bounds to reprojections
    image = collection\
        .toBands()\
        .select(old_band_names, new_band_names)\
        .divide(10000)\
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
    years = sorted(list(set(years)))
    bands = sorted(list(set(bands)))
    xr_list = [
        xr.DataArray(
            np.dstack([arr[f"{y}_{b}"] for b in bands]),
            dims=['y', 'x', "band"]
        ).astype(float)
        for y in years]
    xarr = xr.concat(xr_list, dim=DATETIME_LABEL)
    
    # transposing to match torch convention
    xarr = xarr.transpose("band", DATETIME_LABEL, "x", "y")

    # adding strata data as attributes
    xarr.name = str(index)
    new_attrs = attributes | {"point": point_name, "square": square_name}
    xarr.attrs.update(**new_attrs)

    return xarr.chunk(chunks={DATETIME_LABEL: 1})


def clip_xy_xarray(xarr: xr.DataArray, 
                   pixel_edge_size: int, 
                   buffer_size: int,
                   random_seed: int | None) -> xr.DataArray:
    if buffer_size > 0:
        clip_xy_xarray(xarr, buffer_size*2, 0, None)
        
    x_diff = xarr["x"].size - pixel_edge_size
    y_diff = xarr["y"].size - pixel_edge_size

    if random_seed is None:
        x_start = x_diff // 2 
        x_end = xarr["x"].size - (x_diff - x_start)

        y_start = y_diff // 2
        y_end = xarr["y"].size - (y_diff - y_start)
    else:
        np.random.seed(random_seed)

        x_start = np.random.randint(0, x_diff + 1)
        y_start = np.random.randint(0, y_diff + 1)

        x_end = xarr["x"].size - (x_diff - x_start)
        y_end = xarr["y"].size - (y_diff - y_start)

    return xarr.sel(x=slice(x_start, x_end), y=slice(y_start, y_end))


def pad_xy_xarray(
        xarr: xr.DataArray,
        pixel_edge_size: int) -> xr.DataArray:
    x_diff = pixel_edge_size - xarr["x"].size
    y_diff = pixel_edge_size - xarr["y"].size

    x_start = x_diff // 2
    x_end = x_diff - x_start

    y_start = y_diff // 2
    y_end = y_diff - y_start

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
        look_range: int,
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
    square_coords = list(square.exterior.coords)
    point_coords = list(square.centroid.coords)

    # generating start and end date from datetime attribute and back step
    end_year = int(meta_data.iloc[index].loc[DATETIME_LABEL])

    end_date = datetime(end_year, end_month, end_day)
    start_date = datetime(end_year - look_range, start_month, start_day)

    data_vars = meta_data.iloc[index].to_dict()
    attributes = {k: v for k, v in data_vars.items()
                  if all([s not in k for s in ["geometry", "square", "point"]])}

    return square_coords, \
        point_coords, \
        start_date, \
        end_date, \
        attributes


def function_timer(func):
    logger = get_logger(LOG_PATH, METHOD)

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


def train_validate_test_split(samples: Tuple[float], 
                              ratios: list[int],
                              random_seed: float | int) -> np.array:
    assert len(ratios) == 2 or len(ratios) == 3, "Ratios must be a list or array of 2 ors 3 elements (val, test) or (train, val, test)"
    assert (np.isclose(sum(ratios), 1.0) and len(ratios) == 3) or (sum(ratios) < 1.0 and len(ratios) == 2), "Ratios must sum to 1 if train is included or is < 1 otherwise"

    if len(ratios) == 2:
        ratios = (1 - sum(ratios),) + tuple(ratios)

    n_total = len(samples)
    indices = np.arange(n_total)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_end = int(ratios[0] * n_total)
    val_end = train_end + int(ratios[1] * n_total)

    train = samples[:train_end,...]
    val = samples[train_end:val_end,...]
    test = samples[val_end:,...]

    return train, val, test

@function_timer
def get_xarr_chip_mean_std(data: xr.Dataset) -> tuple[list[float], list[float]]:
    data = data.to_dataarray()
    means = data.mean(dim=["variable", DATETIME_LABEL, "y", "x"])
    stds = data.std(dim=["variable", DATETIME_LABEL, "y", "x"])
    return means.values.tolist(), stds.values.tolist()


@function_timer
def get_xarr_anno_mean_std(data: xr.Dataset) -> tuple[list[float], list[float]]:
    sums = data.sum(dim=["y", "x"]).to_dataarray()
    means = sums.mean(dim="variable")
    stds = sums.std(dim="variable")
    return means.values.tolist(), stds.values.tolist()


@function_timer
def get_class_weights(data: xr.Dataset) -> tuple[list[float], list[float]]:
    sums = data.sum(dim=["y", "x"])
    totals = sums.to_dataarray().sum(dim="variable")
    weights = totals.min() / totals
    return totals.values.tolist(), weights.values.tolist()


@function_timer
def test_non_zero_sum(xarr: xr.Dataset,
                      size: int) -> Generator[tuple[str, int], None, None]:
    tests = np.random.choice(xarr.variables, size=size).tolist()
    for test in tests:
        sum = xarr[test].sum().values
        assert sum > 0
        yield test, int(sum)
