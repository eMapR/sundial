import geopandas as gpd
import pandas as pd
import xarray as xr

from datetime import datetime
from typing import Tuple

from pipeline.settings import DATETIME_LABEL


def parse_meta_data(
        meta_data: pd.DataFrame,
        index: int,
        look_range: int,
        start_month: int,
        start_day: int,
        end_month: int,
        end_day: int) -> Tuple[list[Tuple[float, float]],
                               Tuple[float, float],
                               str,
                               list[Tuple[float, float]],
                               str,
                               datetime | None,
                               datetime | None,
                               dict]:
    square = meta_data.iloc[index].loc["geometry"]
    square_coords = list(square.boundary.coords)
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