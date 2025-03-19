import geopandas as gpd

from datetime import datetime


def get_day_of_year(month_day: str, year: int):
    date_str = f"{year}-{month_day}"
    date = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = date.timetuple().tm_yday
    return day_of_year


# TODO: update these to store and pull from zarr xarray
class LatLonFromMeta():
    meta_data = True
    name = "location_coords"
    
    def get_item(self, img_idx: int, time_indx: int, meta_data: gpd.GeoDataFrame):
        point = meta_data.iloc[img_idx].geometry.centroid
        return [point.y, point.x]


class YearDayFromMeta():
    meta_data = True
    name = "temporal_coords"
    
    def __init__(self,
                 year_col: str,
                 month_day: list[str | datetime]):
        self.year_col = year_col
        self.month_day = month_day
    
    def get_item(self, img_idx: int, time_indx: int, meta_data: gpd.GeoDataFrame):
        year = meta_data[self.year_col].iloc[img_idx]
        return [(year, get_day_of_year(date, year)) for date in self.month_day]


class MultiYearDayFromMeta():
    meta_data = True
    name = "temporal_coords"
    
    def __init__(self,
                 year_col: str,
                 year_range: int,
                 month_day: str,
                 forward: bool):
        self.year_col = year_col
        self.year_range = year_range
        self.month_day = month_day
        self.forward = forward
    
    def get_item(self, img_idx: int, time_indx: int, meta_data: gpd.GeoDataFrame):
        yearbase = meta_data[self.year_col].iloc[img_idx]
        yeardays = []
        
        if self.forward:
            yearbase += 1

        for i in range(self.year_range - 1, -1, -1):
            year = yearbase - 1
            yeardays.append((year, get_day_of_year(self.month_day, year)))

        return yeardays
    

class YearDayFromTimeIndx():
    meta_data = False
    name = "temporal_coords"
    
    def __init__(self,
                 year: int,
                 month_day: str | datetime):
        self.year = year
        self.month_day = month_day
    
    def get_item(self, img_idx: int, time_indx: int):
        return [(self.year, get_day_of_year(date, self.year)) for date in self.month_day]
    

class MultiYearDayFromTimeIndx():
    meta_data = False
    name = "temporal_coords"
    
    def __init__(self,
                 start_year: int,
                 time_step: int,
                 month_day: str | datetime,
                 flip: bool):
        self.start_year = start_year
        self.time_step = time_step
        self.month_day = month_day
        self.flip = flip
    
    def get_item(self, img_idx: int, time_indx: int):
        start = time_indx - self.time_step
        years = [self.start_year + start + y for y in range(self.time_step+1)]
        
        if self.flip:
            years.reverse()
        
        return [(year, get_day_of_year(self.month_day, year)) for year in years]