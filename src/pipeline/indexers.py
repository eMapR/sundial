import numpy as np
import sys
import xarray as xr

from typing import Tuple

from constants import APPEND_DIM, CLASS_LABEL, DATETIME_LABEL, IDX_NAME_ZFILL


def train_validate_test_split(chip_data: xr.DataArray,
                              anno_data: xr.DataArray,
                              ratios: list[int],
                              random_seed: float | int) -> np.array:
    assert len(ratios) == 2 or len(ratios) == 3, "Ratios must be a list or array of 2 ors 3 elements (val, test) or (train, val, test)"
    assert (np.isclose(sum(ratios), 1.0) and len(ratios) == 3) or (sum(ratios) < 1.0 and len(ratios) == 2), "Ratios must sum to 1 if train is included or is < 1 otherwise"

    if len(ratios) == 2:
        ratios = (1 - sum(ratios),) + tuple(ratios)

    n_total = len(chip_data)
    samples = np.arange(n_total)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(samples)

    train_end = int(ratios[0] * n_total)
    val_end = train_end + int(ratios[1] * n_total)

    train = samples[:train_end,...]
    val = samples[train_end:val_end,...]
    test = samples[val_end:,...]

    return train, val, test


def time_window_split(chip_data: xr.DataArray,
                      anno_data: xr.DataArray,
                      ratios: list[int],
                      random_seed: float | int,
                      time_range: Tuple[int],
                      time_step: int,
                      anno_offset: int | None) -> np.array:
    train, val, test = train_validate_test_split(chip_data, anno_data, ratios, random_seed)
    if anno_offset is not None:
        times = np.arange(*time_range, time_step)
        anno_times = times - anno_offset
        train = np.transpose([np.tile(train, len(times)), np.repeat(times, len(train)), np.repeat(anno_times, len(train))])
        val = np.transpose([np.tile(val, len(times)), np.repeat(times, len(val)), np.repeat(anno_times, len(val))])
        test = np.transpose([np.tile(test, len(times)), np.repeat(times, len(test)), np.repeat(anno_times, len(test))])
        
    else:
        times = np.arange(*time_range, time_step)
        train = np.transpose([np.tile(train, len(times)), np.repeat(times, len(train))])
        val = np.transpose([np.tile(val, len(times)), np.repeat(times, len(val))])
        test = np.transpose([np.tile(test, len(times)), np.repeat(times, len(test))])
    
    return train, val, test


def time_window(chip_data: xr.DataArray,
                anno_data: xr.DataArray,
                ratios: list[int],
                random_seed: float | int,
                time_range: Tuple[int],
                time_step: int):
    times = np.arange(*time_range, time_step)
    total = np.arange(len(chip_data))
    sample = np.transpose([np.tile(total, len(times)), np.repeat(times, len(total))])

    return sample


def check_class_sums_helper(class_sums: np.array,
                            class_filters: dict,
                            num_pixels: int):
    total = class_sums.sum()
    if total == 0:
        return False
    
    ratios = class_sums / num_pixels
    checks = []
    
    for class_index in range(len(class_filters)):
        class_filter = class_filters[class_index]
        if class_filter[0] is not None or class_filter[1] is not None:
            ratio = ratios[class_index]
            floor = class_filter[0] if class_filter[0] else -1
            ceiling = class_filter[1] if class_filter[1] else sys.maxsize
            checks.append(ratio >= floor and ratio < ceiling)
    return any(checks)


def check_class_sums(anno_data: xr.DataArray,
                     sample: Tuple[int],
                     class_filters: dict):
    sample_anno = anno_data.sel({APPEND_DIM: sample[0]})
    num_pixels = sample_anno["y"].shape[0]*sample_anno["x"].shape[0]
    dims = ["y", "x"]
    class_sums = sample_anno.sum(dims).values
    if DATETIME_LABEL in sample_anno.dims and len(sample) == 3:
        class_sums = class_sums[sample[2]]
    if check_class_sums_helper(class_sums, class_filters, num_pixels):
        return sample
    else:
        return np.full(sample.shape, np.nan)


def time_window_split_class_filter(chip_data: xr.DataArray,
                                   anno_data: xr.DataArray,
                                   ratios: list[int],
                                   random_seed: float | int,
                                   time_range: Tuple[int],
                                   time_step: int,
                                   anno_offset: int | None,
                                   class_filters: dict,
                                   num_workers: int = 48) -> np.array:
    assert len(class_filters) == anno_data[CLASS_LABEL].shape[0]
    
    splits = time_window_split(chip_data, anno_data, ratios, random_seed, time_range, time_step, anno_offset)
    proc_splits = []
    
    vec_func = lambda t: check_class_sums(anno_data, t, class_filters)
    for split in splits:
        vec = np.array([vec_func(s) for s in split])
        vec = np.where(vec < 0, np.nan, vec) # TODO: negative max value bug
        vec = vec[~np.isnan(vec).any(axis=1)]
        proc_splits.append(vec)
        
    return proc_splits
    
    