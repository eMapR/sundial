import cupy as cp
import geopandas as gpd
import glob
import importlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import torch
import rasterio
import re
import shutil

from cupyx.scipy.ndimage import distance_transform_edt
from rasterio.transform import from_bounds
from sklearn.manifold import TSNE
from typing import Any, Optional


def dynamic_import(loader: dict):
    class_path = loader.get("class_path")
    init_args = loader.get("init_args", {})
    
    loader_path = class_path.rsplit(".", 1)
    module_path, class_name = loader_path
    loader_cls = getattr(importlib.import_module(module_path), class_name)
    
    return loader_cls(**init_args)


def distance_transform(targets: torch.tensor):
    distances = torch.zeros_like(targets, device=targets.device)
    cx = cp.from_dlpack(targets)
    for k in range(targets.shape[0]):
        if cp.any(cx[k] > 0):
            distance = distance_transform_edt(1 - cx[k]) * (1 - cx[k]) \
                - (distance_transform_edt(cx[k]) - 1) * cx[k]
            distances[k] = torch.tensor(distance, dtype=targets.dtype, device=targets.device)
    return distances


def log_tsne_plot(tensor: torch.Tensor,
                  plot_name: str,
                  logger: Any,
                  figsize: tuple[int]) -> None:
    """
    Perform t-SNE on the elements in the first dimension of the tensor
    and save the plot as a PNG file.

    Args:
    - tensor (torch.Tensor): Input tensor
    """
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tensor.cpu().numpy())

    # Plot the results
    plt.figure(figsize=figsize)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    for i in range(tsne_results.shape[0]):
        plt.annotate(f'{i}', (tsne_results[i, 0], tsne_results[i, 1]))
    plt.title(plot_name)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # Save the plot as a PNG file
    logger.log_figure(figure_name=plot_name, figure=plt)


def get_best_ckpt(dir_path: str | os.PathLike,
                  experiment: Optional[str] = None) -> str | None:
    glob_exp = "_" + experiment if experiment else ""
    glob_pat = f"epoch=*_*=*{glob_exp}.ckpt"
    files = glob.glob(os.path.join(dir_path, glob_pat))

    regex_exp = experiment if experiment else ".+"
    regex_str = fr"epoch=(\d+)_(?:.*)=(\d+\.\d+)(?:-v(\d+))?(?:_{regex_exp})?\.ckpt"
    regex = re.compile(regex_str)

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
    return best_file


def get_latest_ckpt(dir_path: str | os.PathLike,
                    experiment: Optional[str] = None) -> str | None:
    glob_exp = "_" + experiment if experiment else ""
    glob_pat = f"epoch=*_*=*{glob_exp}.ckpt"
    files = glob.glob(os.path.join(dir_path, glob_pat))

    regex_exp = experiment if experiment else ".+"
    regex_str = fr"epoch=(\d+)_(?:.*)=(?:\d+\.\d+)(?:-v(\d+))?(?:_{regex_exp})?\.ckpt"
    regex = re.compile(regex_str)

    max_epoch = -1
    current_version = -1
    latest_file = None

    for file in files:
        match = regex.search(file)
        if match:
            epoch, version = match.groups()
            if version is None:
                version = -2
            else:
                version = int(version)
            epoch = int(epoch)

            if epoch > max_epoch or (epoch == max_epoch and version > current_version):
                max_epoch = epoch
                current_version = version
                latest_file = file
    return latest_file


def clean_dir_files(directory: str | os.PathLike):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def save_rgb_ir_tensor(chip: torch.Tensor, index_name: int, path: str):
    times = list(range(chip.shape[1]-1, -1, -1))
    for t in range(chip.shape[1]):
        image = chip[0:3, t, :, :]
        img_path = os.path.join(path, f"{index_name}_rgb_t-{times[t]}_chip.pt")
        torch.save(image, img_path)

        image = chip[3:6, t, :, :]
        img_path = os.path.join(path, f"{index_name}_ir_t-{times[t]}_chip.pt")
        torch.save(image, img_path)


def log_rgb_image(chip: torch.Tensor, index_name: int, label: str, logger: Any, min_sr: float, max_sr: float):
    rgb = chip[0:3].flip(0).permute(1, 2, 3, 0)
    times = list(range(chip.shape[1]-1, -1, -1))
    for t in range(chip.shape[1]):
        image = rgb[t, :, :, :]
        logger.log_image(
            image_data=image.cpu(),
            name=f"{index_name}_rgb_t-{times[t]}_{label}.png",
            image_scale=2.0,
            image_minmax=(min_sr, max_sr)
        )
        

def log_false_color_image(chip: torch.Tensor, index_name: int, label: str, logger: Any, min_sr: float, max_sr: float):
    false_color = chip.index_select(0, torch.tensor([3,2,1], device=chip.device)).permute(1, 2, 3, 0)
    times = list(range(chip.shape[1]-1, -1, -1))
    for t in range(chip.shape[1]):
        image = false_color[t, :, :, :]
        logger.log_image(
            image_data=image.cpu(),
            name=f"{index_name}_false_color_t-{times[t]}_{label}.png",
            image_scale=2.0,
            image_minmax=(min_sr, max_sr)
        )
        
    image =  false_color[1, :, :, :] - false_color[0, :, :, :]
    logger.log_image(
        image_data=image.cpu(),
        name=f"{index_name}_false_color_diff_{label}.png",
        image_scale=2.0,
        image_minmax=(0, max_sr)
    )



def tensors_to_tifs_helper(meta_data: gpd.GeoDataFrame,
                           regex: re.Pattern,
                           output_path: str,
                           file_queue: mp.Queue,
                           logger: Any):
    crs = f"EPSG:{meta_data.crs.to_epsg()}"
    while (file := file_queue.get()) is not None:
        base_name = os.path.basename(file)
        meta_idx = int(regex.search(base_name).groups()[0])
        meta = meta_data.iloc[meta_idx]
        polygon = meta.geometry
        
        tensor = torch.load(file, map_location=torch.device('cpu'), weights_only=False)
        bands = tensor.shape[0]
        minx, miny, maxx, maxy = polygon.bounds
        
        profile = {
            'driver': 'GTiff',
            'height': tensor.shape[1],
            'width': tensor.shape[2],
            'count': bands,
            'dtype': 'float32',
            'crs': crs,
            'transform': from_bounds(minx, miny, maxx, maxy, tensor.shape[2], tensor.shape[1])
        }

        with rasterio.open(os.path.join(output_path, f"{os.path.splitext(base_name)[0]}.tif"), 'w', **profile) as dst:
            dst.write(tensor.numpy())
        logger.info(f"Processed {base_name} at meta {meta_idx}...")


def tensors_to_tifs(prediction_path: str,
                    output_name: str,
                    data_path: str,
                    meta_data_path: str,
                    num_workers: int,
                    logger: Any) -> str:
    output_path = os.path.join(data_path, f"{output_name}_temp")
    if os.path.exists(output_path) and os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    pattern = os.path.join(prediction_path, "*.pt")
    files = glob.glob(pattern)
    meta_data = gpd.read_file(meta_data_path)
    regex = re.compile(r"(?:(\d+)_(?:.+)\.pt)")

    manager = mp.Manager()
    file_queue = manager.Queue()
    file_idxes = []

    for file in files:
        file_queue.put(file)
        logger.info
        file_idxes.append(int(regex.search(file).groups()[0]))

    [file_queue.put(None) for _ in range(num_workers)]
    processes = set()
    for _ in range(num_workers):
        p = mp.Process(
            target=tensors_to_tifs_helper,
            args=(meta_data,
                  regex,
                  output_path,
                  file_queue,
                  logger),
            daemon=True)
        p.start()
        processes.add(p)
    [r.join() for r in processes]

    return output_path
