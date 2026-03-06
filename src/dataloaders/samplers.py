import geopandas as gpd

from datetime import datetime
from shapely.geometry import box

from constants import GEO_PROC_PATH
from pipeline.utils import chunk_bounds, filter_chunks



class DataArraySampler:
    def __init__(split, imagery_da, annotations_da, grid_y_size, grid_x_size):
        self._split = split
        self._imagery_da = imagery_da
        self._annotations_da = annotations_da
        self._geo_proc_data = gpd.read_file(GEO_PROC_PATH)
        
        maxy, miny = self._imagery_da.coords["lat"][0], self._imagery_da.coords["lat"][-1]
        minx, maxx = self._imagery_da.coords["lon"][0], self._imagery_da.coords["lon"][-1]
        self._chunks = chunk_bounds([minx, miny, maxx, maxy], grid_y_size, grid_x_size)
        self._chunks = filter_chunks(chunks, self._geo_proc_data, grid_y_size, grid_x_size)
        
    def __len__(self):
        return len(self._chunks)

    def __call__(self, indx):
        pass