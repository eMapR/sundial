import numpy as np
import xarray as xr

from typing import Optional

from constants import APPEND_DIM, DATETIME_LABEL


def unstack_band_years(
        arr: np.ndarray,
        index: int,
        pixel_edge_size: int,
        square_name: str,
        point_name: str) -> xr.DataArray:

    # unflattening the array to shape (year, band, y, x)
    # TODO: implement np.structured_to_unstructured module
    years, bands = zip(*[b.split('_') for b in arr.dtype.names if b != "overlap"])
    years = sorted(list(set(years)))
    bands = sorted(list(set(bands)))
    xr_list = [
        xr.DataArray(
            np.stack([arr[f"{y}_{b}"] for b in bands]),
            dims=["band", 'y', 'x']
        ).astype(float)
        for y in years]
    xarr = xr.concat(xr_list, dim=DATETIME_LABEL)
    
    # transposing to match torch convention
    xarr = xarr.transpose("band", DATETIME_LABEL, "y", "x")

    # adding sample index and chunking
    xarr = xarr.chunk(chunks={DATETIME_LABEL: 1})
    xarr = xarr.assign_coords({APPEND_DIM: index})

    return xarr