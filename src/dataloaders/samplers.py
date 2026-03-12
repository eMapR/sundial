import geopandas as gpd
import numpy as np
import torch

from datetime import datetime
from shapely import STRtree
from shapely.geometry import box

from constants import GEO_PROC_PATH
from pipeline.settings import PIPELINE_CONFIG
from pipeline.utils import chunk_bounds


class DataArraySampler:
    def __init__(self, imagery_da, annotations_da, split, grid_y_size, grid_x_size, pixel_size, class_indices, offset, window=(2, 2)):
        self._imagery_da = imagery_da
        self._annotations_da = annotations_da
        self._split = split
        self._grid_y_size = grid_y_size
        self._grid_x_size = grid_x_size
        self._pixel_size = pixel_size
        self._class_indices = class_indices
        self._offset = offset
        self._window = window

        self._geo_proc_data = gpd.read_file(GEO_PROC_PATH)
        self._date_column = PIPELINE_CONFIG.get("annotator")["init_args"]["date_column"]
        self._label_column = PIPELINE_CONFIG.get("annotator")["init_args"]["label_column"]
        self._years = sorted(self._geo_proc_data[self._date_column].unique())
        self._labels = sorted(self._geo_proc_data[self._label_column].unique())
        self._labels = [v for i, v in enumerate(self._labels) if i in self._class_indices]
        self._geo_proc_data = self._geo_proc_data.loc[self._geo_proc_data[self._label_column].isin(self._labels)]
            
        maxy, miny = self._imagery_da.coords["lat"][0], self._imagery_da.coords["lat"][-1]
        minx, maxx = self._imagery_da.coords["lon"][0], self._imagery_da.coords["lon"][-1]
        self._chunks = chunk_bounds([minx, miny, maxx, maxy], grid_y_size, grid_x_size)
        self._chunk_samples = []
        
        spatial_index = STRtree(self._geo_proc_data.geometry)
        for chunk in self._chunks:
            ty, tx = chunk
            bounds = tx, ty - grid_y_size, tx + grid_x_size, ty
            hits = spatial_index.query(box(*bounds), predicate="intersects")
            if hits.size > 0:
                for year in np.sort(self._geo_proc_data.iloc[hits][self._date_column].unique()):
                    self._chunk_samples.append((bounds, year))
        
    def __len__(self):
        return len(self._chunk_samples)

    def __call__(self, indx):
        (tx, tY, tX, ty), year = self._chunk_samples[indx]

        ydx = self._years.index(year)
        idx = ydx+self._offset
        time_indices = list(range(idx - self._window[0], idx + self._window[1]))

        imagery = self._imagery_da.sel(
            lat=slice(ty, tY + self._pixel_size),
            lon=slice(tx, tX - self._pixel_size)
        ).isel(time=time_indices).to_numpy()
        imagery = torch.tensor(imagery)
        imagery = torch.where(imagery.isnan(), -1.0, imagery)

        annotation = self._annotations_da.sel(
            lat=slice(ty, tY + self._pixel_size),
            lon=slice(tx, tX - self._pixel_size)
        ).isel(band=self._class_indices, time=ydx).to_numpy()
        annotation = torch.tensor(annotation)
        annotation = torch.where(annotation.isnan(), 0, annotation)
        
        
        return {"chip": imagery,
                "anno": annotation,
                "meta": {
                    "bounds": (tx, tY, tX, ty),
                    "idx": idx,
                    "ydx": ydx}
                }
        