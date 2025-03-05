import geopandas as gpd

from datetime import datetime


# TODO: update these to store and pull from zarr xarray
class LatLonFromMeta():
    meta_data = True
    name = "location_coords"
    
    def get_item(self, idx: int, meta_data: gpd.GeoDataFrame):
        point = meta_data.iloc[idx].geometry.centroid
        return [point.y, point.x]

class YearDayFromMeta():
    meta_data = True
    name = "temporal_coords"
    
    def __init__(self,
                 year_col: str,
                 dates: list[str | datetime]):
        self.year_col = year_col
        self.dates = dates
    
    def get_day_of_year(self, month_day: str, year: int):
        date_str = f"{year}-{month_day}"
        date = datetime.strptime(date_str, "%Y-%m-%d")

        day_of_year = date.timetuple().tm_yday
        return day_of_year
    
    def get_item(self, idx: int, meta_data: gpd.GeoDataFrame):
        year = meta_data[self.year_col].iloc[idx]
        return [(year, self.get_day_of_year(date, year)) for date in self.dates]
    

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
    
    def get_item(self, idx: int, meta_data: gpd.GeoDataFrame):
        yearbase = meta_data[self.year_col].iloc[idx]
        yeardays = []
        
        if self.forward:
            yearbase += 1

        for i in range(self.year_range - 1, -1, -1):
            year = yearbase - 1
            date_str = f"{year}-{self.month_day}"
            date = datetime.strptime(date_str, "%Y-%m-%d")
            day_of_year = date.timetuple().tm_yday
            yeardays.append((year, day_of_year))

        return yeardays