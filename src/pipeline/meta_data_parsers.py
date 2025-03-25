import geopandas as gpd
import pandas as pd
import xarray as xr

from datetime import datetime

from constants import DATETIME_LABEL


def medoid_from_year(
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

    # generating start and end date from datetime attribute and back step
    end_year = int(meta_data.iloc[index].loc[DATETIME_LABEL])

    end_date = datetime(end_year, end_month, end_day)
    start_date = datetime(end_year - look_range, start_month, start_day)

    return square, \
        start_date, \
        end_date