import geopandas as gpd
import glob
import multiprocessing as mp
import os
import torch
import rasterio
import re
import shutil

from typing import Optional, Any

from pipeline.utils import function_timer


def get_best_ckpt(dir_path: str | os.PathLike,
                  experiment: Optional[str] = None) -> str | None:
    glob_exp = experiment if experiment else ""
    glob_pat = f"epoch-*_val_loss-*_{glob_exp}.ckpt"
    files = glob.glob(os.path.join(dir_path, glob_pat))

    regex_exp = experiment if experiment else ".+"
    regex_str = fr"epoch-(\d+)_val_loss-(\d+\.\d+)(?:-v(\d+))?(?:_{regex_exp})?\.ckpt"
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
    glob_exp = experiment if experiment else ""
    glob_pat = f"epoch-*_val_loss-*_{glob_exp}.ckpt"
    files = glob.glob(os.path.join(dir_path, glob_pat))

    regex_exp = experiment if experiment else ".+"
    regex_str = fr"epoch-(\d+)_val_loss-(?:\d+\.\d+)(?:-v(\d+))?(?:_{regex_exp})?\.ckpt"
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


def save_rgb_ir_tensor(chip: torch.Tensor, index: int, path: str):
    times = list(range(chip.shape[1]-1, -1, -1))
    for t in range(chip.shape[1]):
        image = chip[0:3, t, :, :]
        img_path = os.path.join(path, f"{index:07d}_rgb_t-{times[t]}_chip.pt")
        torch.save(image, img_path)

        image = chip[3:6, t, :, :]
        img_path = os.path.join(path, f"{index:07d}_ir_t-{times[t]}_chip.pt")
        torch.save(image, img_path)


def log_rbg_ir_image(chip: torch.Tensor, index: int, logger: Any):
    rgb = chip[0:3].flip(0).permute(1, 2, 3, 0)
    ir = chip[3:6].permute(1, 2, 3, 0)
    rgb_max = torch.max(rgb).item()
    rgb_min = torch.min(rgb).item()
    ir_max = torch.max(ir).item()
    ir_min = torch.min(ir).item()
    times = list(range(chip.shape[1]-1, -1, -1))
    for t in range(chip.shape[1]):
        image = rgb[t, :, :, :]
        logger.log_image(
            image_data=image.detach().cpu(),
            name=f"{index:07d}_rgb_t-{times[t]}_chip",
            image_scale=2.0,
            image_minmax=(rgb_min, rgb_max)
        )
        image = ir[t, :, :, :]
        logger.log_image(
            image_data=image.detach().cpu(),
            name=f"{index:07d}_ir_t-{times[t]}_chip",
            image_scale=2.0,
            image_minmax=(ir_min, ir_max)
        )


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
                    output_name: str,
                    meta_data_path: str,
                    num_workers: int = 1) -> str:
    output_path = os.path.join("/tmp", output_name)
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
    return output_path
