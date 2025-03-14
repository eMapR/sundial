import numpy as np
import sys
import xarray as xr

from typing import Tuple

from pipeline.settings import IDX_NAME_ZFILL


def train_validate_test_split(chip_data: xr.Dataset,
                              anno_data: xr.Dataset,
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


def time_window_split(chip_data: xr.Dataset,
                      anno_data: xr.Dataset,
                      ratios: list[int],
                      random_seed: float | int,
                      time_range: Tuple[int],
                      time_step: int) -> np.array:
    train, val, test = train_validate_test_split(chip_data, anno_data, ratios, random_seed)
    times = np.arange(*time_range, time_step)
    
    train = np.transpose([np.tile(train, len(times)), np.repeat(times, len(train))])
    val = np.transpose([np.tile(val, len(times)), np.repeat(times, len(val))])
    test = np.transpose([np.tile(test, len(times)), np.repeat(times, len(test))])
    
    return train, val, test


def check_class_sums_helper(class_sums: np.array,
                            class_filters: dict,
                            num_pixels: int):
    total = class_sums.sum()
    if total == 0:
        return False
    
    ratios = class_sums / num_pixels
    for class_index in range(len(class_filters)):
        class_filter = class_filters[class_index]
        ratio = ratios[class_index]
        
        floor = class_filter[0] if class_filter[0] else -1
        ceiling = class_filter[1] if class_filter[1] else sys.maxsize
        if floor <= ratio < ceiling:
            pass
        else:
            return False
    return True


def check_class_sums(anno_data: xr.Dataset,
                     sample: Tuple[int],
                     time_step: int,
                     class_filters: dict):
    sample_anno = anno_data[str(sample[0]).zfill(IDX_NAME_ZFILL)]
    num_pixels = sample_anno.shape[-2]*sample_anno.shape[-1]
    class_sums = np.array(sample_anno.attrs["class_sums"])
    class_sums = class_sums[sample[1] - time_step]
    
    if check_class_sums_helper(class_sums, class_filters, num_pixels):
        return sample
    else:
        return np.array([np.nan, np.nan])


def time_window_split_class_filter(chip_data: xr.Dataset,
                                   anno_data: xr.Dataset,
                                   ratios: list[int],
                                   random_seed: float | int,
                                   time_range: Tuple[int],
                                   time_step: int,
                                   class_filters: dict,
                                   num_workers: int = 48) -> np.array:
    splits = time_window_split(chip_data, anno_data, ratios, random_seed, time_range, time_step)
    proc_splits = []
    
    vec_func = lambda t: check_class_sums(anno_data, t, time_step, class_filters)
    for split in splits:
        vec = np.apply_along_axis(vec_func, axis=1, arr=split)
        vec = np.where(vec < 0, np.nan, vec) # TODO: negative max value bug
        vec = vec[~np.isnan(vec).any(axis=1)]
        proc_splits.append(vec)
        
    return proc_splits
    
    