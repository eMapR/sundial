import geopandas as gpd
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from datetime import datetime
from shapely import STRtree
from shapely.geometry import box

from constants import GEO_PROC_PATH
from pipeline.settings import PIPELINE_CONFIG
from pipeline.utils import chunk_bounds


class DataArraySampler:
    def __init__(self,
                 imagery_da: xr.DataArray,
                 annotations_da: xr.DataArray,
                 split: str, 
                 grid_y_size: int,
                 grid_x_size: int,
                 pixel_size: int,
                 class_indices: list[int],
                 toffset: int,
                 window: tuple[int, int]=(2, 2),
                 jitters: int=None,
                 jitter_size: int=None):
        self._imagery_da = imagery_da
        self._annotations_da = annotations_da
        self._split = split
        self._grid_y_size = grid_y_size
        self._grid_x_size = grid_x_size
        self._pixel_size = pixel_size
        self._class_indices = class_indices
        self._toffset = toffset
        self._window = window
        self._jitters = jitters
        self._jitter_size = jitter_size

        self._geo_proc_data = gpd.read_file(GEO_PROC_PATH)
        annotator_kwargs = PIPELINE_CONFIG.get("annotator", {}).get("init_args", {})
        self._date_column = annotator_kwargs.get("date_column",)
        self._label_column = annotator_kwargs.get("label_column")

        if self._date_column is not None:
            self._times = sorted(self._geo_proc_data[self._date_column].unique())
        else:
            self._times = list(range(self._imagery_da['time'].shape[0]-(sum(self._window)-1)))

        if self._label_column is not None:    
            self._labels = sorted(self._geo_proc_data[self._label_column].unique())
            self._labels = [v for i, v in enumerate(self._labels) if i in self._class_indices]
            self._geo_proc_data = self._geo_proc_data.loc[self._geo_proc_data[self._label_column].isin(self._labels)]
            
        self.maxy, self.miny = self._imagery_da.coords["lat"][0].item(), self._imagery_da.coords["lat"][-1].item()
        self.minx, self.maxx = self._imagery_da.coords["lon"][0].item(), self._imagery_da.coords["lon"][-1].item()
        self._chunks = chunk_bounds([self.minx, self.miny, self.maxx, self.maxy], grid_y_size, grid_x_size)
        self._chunk_samples = []
        
        spatial_index = STRtree(self._geo_proc_data.geometry)
        for chunk in self._chunks:
            tY, tx = chunk
            bounds = tx, tY - grid_y_size, tx + grid_x_size, tY
            hits = spatial_index.query(box(*bounds), predicate="intersects")
            if hits.size > 0:
                if self._date_column is not None:
                    for time in self._geo_proc_data.iloc[hits][self._date_column].unique():
                        self._chunk_samples.append((bounds, time))
                else:
                    for time in self._times:
                        self._chunk_samples.append((bounds, time))

    def __len__(self):
        if self._jitters is not None:
            return len(self._chunk_samples)*self._jitters
        else:
            return len(self._chunk_samples)

    def __call__(self, indx):
        out = {}
        
        if self._jitters is not None:
            sindx = indx % len(self._chunk_samples)
            (tx, ty, tX, tY), time = self._chunk_samples[sindx]

            y_rnd = list(range(self._jitter_size-self._grid_y_size, self._grid_y_size-self._jitter_size, self._jitter_size))
            y_jit = y_rnd[np.random.randint(len(y_rnd))]
            x_rnd = list(range(self._jitter_size-self._grid_x_size, self._grid_x_size-self._jitter_size, self._jitter_size))
            x_jit = x_rnd[np.random.randint(len(x_rnd))]
            tx, ty, tX, tY = tx+x_jit, ty+y_jit, tX+x_jit, tY+y_jit
        else:
            (tx, ty, tX, tY), time = self._chunk_samples[indx]            

        tdx = self._times.index(time)
        idx = tdx+self._toffset
        time_indices = list(range(idx - self._window[0], idx + self._window[1]))
        out["meta"] = {"bounds": (tx, ty, tX, tY),
                       "idx": idx,
                       "tdx": tdx,
                       "ypad": (0,0,0,0),
                       "xpad": (0,0,0,0)}

        ymin, ymax = max(self.miny, ty+self._pixel_size), min(tY, self.maxy)
        xmin, xmax = max(self.minx, tx), min(tX-self._pixel_size, self.maxx)
        yslc = slice(ymax, ymin)
        xslc = slice(xmin, xmax)

        imagery = self._imagery_da.sel(
            lat=yslc,
            lon=xslc
        ).isel(time=time_indices).to_numpy()
        imagery = torch.tensor(imagery)
        imagery = torch.where(imagery.isnan(), -1.0, imagery)

        H, W = imagery.shape[-2], imagery.shape[-1]
        ydiff = self._grid_y_size//self._pixel_size - H
        xdiff = self._grid_x_size//self._pixel_size - W
       
        if ydiff > 0:
            ypad = (0, 0, ydiff, 0) if ymax == self.maxy else (0, 0, 0, ydiff)
            imagery = F.pad(imagery, ypad, mode='constant', value=-1)
            out["meta"]["ypad"] = ypad
        if xdiff > 0:
            xpad = (xdiff, 0, 0, 0) if xmin == self.minx else (0, xdiff, 0, 0)
            imagery = F.pad(imagery, xpad, mode='constant', value=-1)
            out["meta"]["xpad"] = xpad
            
        out["inpt"] = imagery

        if self._annotations_da is not None:
            annotation = self._annotations_da.sel(
                lat=yslc,
                lon=xslc
            ).isel(band=self._class_indices, time=tdx).to_numpy()
            annotation = torch.tensor(annotation)
            annotation = torch.where(annotation.isnan(), 0, annotation)
            out["target"] = annotation
            
            if ydiff > 0:
                annotation = F.pad(annotation, ypad, mode='constant', value=0)
            if xdiff > 0:
                annotation = F.pad(annotation, xpad, mode='constant', value=0)
            
        return out


class PrePostPixelPatchSampler:
    def __init__(self,
                 imagery_da: xr.DataArray, 
                 split: str, 
                 epoch_sample_ratio: float,
                 sample_size: int,
                 pixel_size: int,
                 temporal_size:int=16, temporal_holdout: int=3, cache: bool=False, **kwargs):
        self._imagery_da = imagery_da
        self._split = split
        self._epoch_sample_ratio = epoch_sample_ratio
        self._sample_size = sample_size
        self._pixel_size = pixel_size
        self._temporal_size = temporal_size
        self._temporal_holdout = temporal_holdout
        
        self._patch_size = sample_size*pixel_size
        
        y_coords = np.meshgrid(self._imagery_da.coords["lat"][-self._sample_size:])
        x_coords = np.meshgrid(self._imagery_da.coords["lon"][-self._sample_size:])
        self._all_pixels = np.array(np.meshgrid(y_coords, x_coords)).T.reshape(-1, 2)
        
        self._num_pixels = int(self._all_pixels*self._epoch_sample_ratio)
        self.resample()
        
        if cache:
            self._imagery_da = self._imagery_da.compute()
        
    def __len__(self):
        return self._num_pixels

    def __call__(self, indx):
        corner = self._pixels[indx]
        ty, tx = corner
        tY, tX = ty - self._patch_size, tx + self._patch_size
        
        imagery = self._imagery_da.sel(
            lat=slice(ty, tY + self._pixel_size),
            lon=slice(tx, tX - self._pixel_size)
        )
        T = self._imagery_da.shape[1]
    
        if self.split in ["train", "validate"]:
            tindx = np.random.randint(self._temporal_size+self._temporal_holdout, T-self._temporal_holdout)
            cur = imagery.isel(time=slice(tindx-self._temporal_size, tindx)).to_numpy()
            cur = torch.tensor(cur)
            cur = torch.where(cur.isnan(), -1.0, cur)
            cur = cur.unsqueeze(1)
            bwd = []
            for i in range(self._temporal_holdout, 0, -1):
                bindx = tindx-i
                p = imagery.isel(time=slice(bindx-self._temporal_size, bindx)).to_numpy()
                p = torch.tensor(p)
                p = torch.where(p.isnan(), -1.0, p)
                bwd.append(p)
            bwd = torch.stack(bwd, dim=1)
            
            fwd = []
            for i in range(1, self._temporal_holdout+1):
                findx = tindx+i
                p = imagery.isel(time=slice(findx-self._temporal_size, findx)).to_numpy()
                p = torch.tensor(p)
                p = torch.where(p.isnan(), -1.0, p)
                fwd.append(p)
            fwd = torch.stack(fwd, dim=1)
        
            inpt = torch.concat([bwd,cur,fwd], dim=1).flatten(start_dim=1, end_dim=2)
            return {"inpt": inpt,
                    "meta": {"bounds": (ty, tx)}}
        else:
            curs = []
            for tindx in range(self._temporal_size, self._imagery_da.shape[1]):
                cur = imagery.isel(time=slice(tindx-self._temporal_size, tindx)).to_numpy()
                cur = torch.tensor(cur)
                cur = torch.where(cur.isnan(), -1.0, imagery)
                curs.append(cur)
            inpt = torch.stack(curs, dim=1).flatten(start_dim=1, end_dim=2)
            return {"inpt": inpt,
                    "meta": {"point": (ty-self._pixel_size, tx+self._pixel_size)}}

    
    def resample(self):
        self._pixels = self._all_pixels[np.random.choice(np.arange(len(self._pixels)), size=self._num_pixels, replace=False)]


class SinglePrePostPixelPatchSampler:
    def __init__(self,
                 imagery_da: xr.DataArray, 
                 split: str, 
                 point_path: str,
                 sample_size: int,
                 pixel_size: int,
                 temporal_size:int=16, cache: bool=False, **kwargs):
        self._imagery_da = imagery_da
        self._split = split
        self._point_path = point_path
        self._sample_size = sample_size
        self._pixel_size = pixel_size
        self._temporal_size = temporal_size
        self._patch_buf = (sample_size//2)*pixel_size
        
        from scipy.spatial import cKDTree
        pairs = np.array(np.meshgrid(self._imagery_da.coords["lat"], self._imagery_da.coords["lon"])).T.reshape(-1, 2)
        tree = cKDTree(pairs)
        
        self._geo_proc_data = gpd.read_file(self._point_path)

        queries = np.array(self._geo_proc_data.geometry.centroid.map(lambda g: list(g.coords[0])).to_numpy().tolist())
        queries = np.flip(queries, axis=1)
        distances, closest_idx = tree.query(queries)
        
        self._center_points = pairs[closest_idx]

        if cache:
            self._imagery_da = self._imagery_da.compute()
        
    def __len__(self):
        return len(self._center_points)

    def __call__(self, indx):
        ty, tx = self._center_points[indx % len(self._geo_proc_data)]
        b = self._patch_buf
        curs = []
        for tindx in range(self._temporal_size, self._imagery_da.shape[1]):
            imagery = self._imagery_da.sel(
                lat=slice(ty+b, ty-b),
                lon=slice(tx-b, tx+b)
            ).isel(time=slice(tindx-self._temporal_size, tindx)).to_numpy()
            imagery = torch.tensor(imagery)
            imagery = torch.where(imagery.isnan(), -1.0, imagery)
            curs.append(imagery)
        
        return {"inpt": torch.stack(curs, dim=1).flatten(start_dim=1, end_dim=2),
                "meta": {"point": (ty, tx)}}
        