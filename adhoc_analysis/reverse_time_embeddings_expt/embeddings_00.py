import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import xarray as xr

from scipy.stats import gaussian_kde, linregress
from typing import Tuple


CLASS_IDX = 6
TIME_STEP = 1
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
EXPERIMENT_SUFFIX = os.getenv("EXPERIMENT_SUFFIX")
BASE_PATH = os.getenv("BASE_PATH")
EXPERIMENT_PATH = os.path.join(BASE_PATH, EXPERIMENT_SUFFIX)

def check_class_sums(chip_data: xr.DataArray,
                     anno_data: xr.DataArray,
                     samples,
                     unfold,
                     sumpool,):
    
    selected_ndvi_t0 = []
    selected_ndvi_t1 = []
    selected_band_t0 = []
    selected_band_t1 = []
    class_sums_pool = []
    
    for sample in samples:
        samp_idx = int(sample[1])
        anno_idx = samp_idx - TIME_STEP

        anno = anno_data.sel(sample=sample[0]).isel({'datetime': anno_idx, 'class': CLASS_IDX}).values
        harv = sumpool(torch.tensor(anno).unsqueeze(0)).flatten()
        condition = (harv > 0)
        ann_indices = torch.where(condition)[0]
        class_sums_pool.append(harv[ann_indices])
        chip = chip_data.sel(sample=sample[0]).isel({'datetime': slice(samp_idx-TIME_STEP, samp_idx+1)}).values
        
        nir = chip[3, :, :, :]
        red = chip[2, :, :, :]
        
        # be warned my friends nan values are a coming'
        ndvi = torch.tensor((nir - red) / (nir + red)).unsqueeze(1)
        ndvi = unfold(ndvi)
        
        chip = torch.tensor(chip).permute(1, 0, 2, 3)
        band = unfold(chip)

        selected_ndvi_t0.append(ndvi[0, :, ann_indices])
        selected_ndvi_t1.append(ndvi[1, :, ann_indices])
        selected_band_t0.append(band[0, :, ann_indices])
        selected_band_t1.append(band[1, :, ann_indices])
        
    class_sums_pool = torch.concat(class_sums_pool, dim=0)
    selected_ndvi_t0 = torch.concat(selected_ndvi_t0, dim=1).transpose(0,1)
    selected_ndvi_t1 = torch.concat(selected_ndvi_t1, dim=1).transpose(0,1)
    selected_band_t0 = torch.concat(selected_band_t0, dim=1).transpose(0,1)
    selected_band_t1 = torch.concat(selected_band_t1, dim=1).transpose(0,1)
    
    torch.save(class_sums_pool, os.path.join(EXPERIMENT_PATH, "class_sums_pool.pt"))
    torch.save(selected_ndvi_t0, os.path.join(EXPERIMENT_PATH, "ndvi_t0.pt"))
    torch.save(selected_ndvi_t1, os.path.join(EXPERIMENT_PATH, "ndvi_t1.pt"))
    torch.save(selected_band_t0, os.path.join(EXPERIMENT_PATH, "band_t0.pt"))
    torch.save(selected_band_t1, os.path.join(EXPERIMENT_PATH, "band_t1.pt"))

    return selected_ndvi_t0, selected_ndvi_t1, class_sums_pool

if __name__ == "__main__":

    chip_data = xr.open_dataarray(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/chip_data", engine="zarr")
    anno_data = xr.open_dataarray(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/anno_data", engine="zarr")
    filtered = np.load(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/train_sample.npy")
    unfold = torch.nn.Unfold(kernel_size=(16,16), stride=(16,16))
    sumpool = torch.nn.AvgPool2d(16, divisor_override=1)
    
    ndvi_t0, ndvi_t1, class_sums_pool = check_class_sums(chip_data,
                                        anno_data,
                                        filtered,
                                        unfold,
                                        sumpool)
    
    ndvi_t0 = torch.mean(ndvi_t0, dim=1)
    ndvi_t1 = torch.mean(ndvi_t1, dim=1)
    
    combined_mask = torch.isnan(ndvi_t0) | torch.isnan(ndvi_t1)

    xy = np.vstack([class_sums_pool[~combined_mask], ndvi_t0[~combined_mask]])
    z = gaussian_kde(xy)(xy)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    def plot_with_regression(ax, x, y, title, mn=-1, mx=1):
        ax.scatter(x, y, c=z, alpha=0.7, s=1)
        ax.set_title(title)
        ax.set_xlabel("# px of Δ in 16 x 16 patch")
        ax.set_ylabel("NDVI")
        ax.set_xlim(0,256)
        ax.set_ylim(mn,mx)

    plot_with_regression(axes[0], class_sums_pool[~combined_mask], ndvi_t0[~combined_mask], 'Timestep 0')
    plot_with_regression(axes[1], class_sums_pool[~combined_mask], ndvi_t1[~combined_mask], 'Timestep 1')
    plot_with_regression(axes[2], class_sums_pool[~combined_mask], ndvi_t1[~combined_mask]-ndvi_t0[~combined_mask], 'NDVI Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, "ndvi.png"), dpi=300)

    
