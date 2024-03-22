import geopandas as gpd
import glob
import multiprocessing as mp
import os
import torch
import rasterio
import re

from pipeline.utils import function_timer


def get_best_ckpt(dir_path):
    pattern = "*epoch-*_val_loss-*.ckpt"
    regex = re.compile(
        r"(?:.+_)?epoch-(\d+)_val_loss-(\d+\.\d+)(?:-v(\d+))?\.ckpt")
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


def tensors_to_tifs_helper(meta_data: gpd.GeoDataFrame,
                           regex: re.Pattern,
                           output_path: str,
                           file_queue: mp.Queue):
    crs = f"EPSG:{meta_data.crs.to_epsg()}"
    while (file := file_queue.get()) is not None:
        meta_idx = int(regex.search(file).groups()[0])
        meta = meta_data.iloc[meta_idx]
        polygon = meta.geometry

        tensor = torch.load(file, map_location=torch.device('cpu'))
        base_name = os.path.splitext(os.path.basename(file))[0]
        bands = tensor.shape[0]
        minx, miny, maxx, maxy = polygon.bounds

        profile = {
            'driver': 'GTiff',
            'height': tensor.shape[1],
            'width': tensor.shape[2],
            'count': bands,
            'dtype': 'float32',
            'crs': crs,
            'transform': rasterio.transform.from_bounds(minx, miny, maxx, maxy, tensor.shape[2], tensor.shape[1])
        }

        with rasterio.open(os.path.join(output_path, f"{base_name}.tif"), 'w', **profile) as dst:
            dst.write(tensor.numpy())


@function_timer
def tensors_to_tifs(prediction_path: str,
                    output_path: str,
                    meta_data_path: str,
                    num_workers: int = 1):
    pattern = os.path.join(prediction_path, "*.pt")
    files = glob.glob(pattern)
    meta_data = gpd.read_file(meta_data_path)
    regex = re.compile(r"(?:(\d+)_(?:.+)\.pt)")

    manager = mp.Manager()
    file_queue = manager.Queue()
    file_idxes = []

    for file in files:
        file_queue.put(file)
        file_idxes.append(int(regex.search(file).groups()[0]))

    [file_queue.put(None) for _ in range(num_workers)]
    processes = set()
    for _ in range(num_workers):
        p = mp.Process(
            target=tensors_to_tifs_helper,
            args=(meta_data,
                  regex,
                  output_path,
                  file_queue),
            daemon=True)
        p.start()
        processes.add(p)
    [r.join() for r in processes]

    meta = meta_data.iloc[file_idxes].reset_index()
    meta.to_file(os.path.join(output_path, "meta_data"))