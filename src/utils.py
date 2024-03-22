import geopandas as gpd
import glob
import multiprocessing as mp
import os
import torch
import rasterio
import re

from shapely.geometry import Polygon


def get_best_ckpt(dir_path):
    pattern = "epoch-*_val_loss-*.ckpt"
    regex = re.compile(r"epoch-(\d+)_val_loss-(\d+\.\d+)(?:-v(\d+))?\.ckpt")
    files = glob.glob(os.path.join(dir_path, pattern))

    min_val_loss = float('inf')
    current_epoch = -1
    current_version = -1
    best_file = None

    for file in files:
        match = regex.search(file)
        if match:
            epoch, val_loss, version = match.groups()
            if version is None:
                version = -2
            else:
                version = int(version)
            epoch = int(epoch)
            val_loss = float(val_loss)

            if val_loss < min_val_loss or (val_loss == min_val_loss and (epoch > current_epoch or (epoch == current_epoch and version > current_version))):
                min_val_loss = val_loss
                current_epoch = int(epoch)
                current_version = version
                best_file = file
    if best_file is not None:
        return best_file
    else:
        raise FileNotFoundError("No checkpoint found in the directory.")


def tensors_to_tifs_helper(polygon: Polygon,
                           file: str,
                           crs: str,
                           regex: re.Pattern,
                           output_path: str):
    index = int(regex.search(file).groups()[0])
    tensor = torch.load(file)
    minx, miny, maxx, maxy = polygon.bounds

    profile = {
        'driver': 'GTiff',
        'height': tensor.shape[1],
        'width': tensor.shape[2],
        'count': 1,
        'dtype': 'float32',
        'crs': crs,
        'transform': rasterio.transform.from_bounds(minx, miny, maxx, maxy, tensor.shape[2], tensor.shape[1])
    }

    with rasterio.open(os.path.join(output_path, f"{index}_pred.tif"), 'w', **profile) as dst:
        dst.write(tensor.numpy(), 1)


def tensors_to_tifs(prediction_path: str,
                    output_path: str,
                    meta_data_path: str,
                    num_workers: int = 1):
    pattern = os.path.join(prediction_path, "*.pt")
    regex = re.compile(r"(?:(\d+)_pred\.pt)")
    files = glob.glob(pattern)
    meta_data = gpd.read_file(meta_data_path)
    epsg_str = f"EPSG:{meta_data.crs.to_epsg()}"

    with mp.Pool(processes=num_workers) as pool:
        pool.starmap(lambda p, f: tensors_to_tifs_helper(
            p, f, epsg_str, regex, output_path), zip(meta_data.geometry, files))
