import numpy as np
import xarray as xr

from typing import Optional

from pipeline.settings import DATETIME_LABEL


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