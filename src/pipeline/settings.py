import os
import yaml

from datetime import date


def save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# sample information
SAMPLE_NAME = os.getenv("SUNDIAL_SAMPLE_NAME")
EXPERIMENT_PREFIX = os.getenv("SUNDIAL_EXPERIMENT_PREFIX")
EXPERIMENT_NAME = os.getenv("SUNDIAL_EXPERIMENT_NAME")
METHOD = os.getenv("SUNDIAL_METHOD")

# base paths
BASE_PATH = os.getenv("SUNDIAL_BASE_PATH")
CONFIGS_PATH = os.path.join(BASE_PATH, "configs")
SHAPES_PATH = os.path.join(BASE_PATH, "shapes")
SAMPLES_PATH = os.path.join(BASE_PATH, "samples")
CHECKPOINTS_PATH = os.path.join(BASE_PATH, "checkpoints")
PREDICTIONS_PATH = os.path.join(BASE_PATH, "predictions")
LOGS_PATH = os.path.join(BASE_PATH, "logs")

# experiment paths
CONFIG_PATH = os.path.join(CONFIGS_PATH, EXPERIMENT_NAME)
SAMPLE_PATH = os.path.join(SAMPLES_PATH, EXPERIMENT_NAME)
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_PATH, EXPERIMENT_NAME)
PREDICTION_PATH = os.path.join(PREDICTIONS_PATH, EXPERIMENT_NAME)
LOG_PATH = os.path.join(LOGS_PATH, EXPERIMENT_NAME)

# config paths
SAMPLE_CONFIG_PATH = os.path.join(CONFIG_PATH, "sample.yaml")

# sample and data paths
META_DATA_PATH = os.path.join(SAMPLE_PATH, "meta_data")
STRATA_MAP_PATH = os.path.join(SAMPLE_PATH, "strata_map.yaml")
STAT_DATA_PATH = os.path.join(SAMPLE_PATH, "stat_data.yaml")
CHIP_DATA_PATH = os.path.join(SAMPLE_PATH, "chip_data.zarr")
ANNO_DATA_PATH = os.path.join(SAMPLE_PATH, "anno_data.zarr")
TRAIN_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "train_sample.npy")
VALIDATE_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "validate_sample.npy")
PREDICT_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "predict_sample.npy")
TEST_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "test_sample.npy")

# zipped shapefile and source data paths
GEO_RAW_PATH = os.path.join(SHAPES_PATH, SAMPLE_NAME)
GEO_PRE_PATH = os.path.join(SAMPLE_PATH, "geo_file")

# image and meta data settings
RANDOM_STATE = 42
MASK_LABELS = ["cloud"]
STRATA_ATTR_NAME = "stratum"
GEE_REQUEST_LIMIT = 40
NO_DATA_VALUE = 0

# GEE file type settings
FILE_EXT_MAP = {
    "GEO_TIFF": "tif",
    "NPY": "npy",
    "NUMPY_NDARRAY": "npy",
    "ZARR": "zarr"
}

SAMPLER = {
    # sampling settings
    "sample_toggles": {
        "preprocess_data": True,
        "stratify_data": True,
        "generate_squares": True,
        "generate_time_combinations": False,
        "generate_train_test_splits": True,
    },

    # date information for medoid composites
    "medoid_config": {
        "start_month": 7,
        "start_day": 15,
        "end_month": 9,
        "end_day": 1
    },

    # sampling settings
    # (str) method to use for sampling
    "method": "centroid",
    # (float) fraction of total points to sample to pull from
    "fraction": 2.0e-2,
    # (int | None) number of points to sample. This is used if fraction is None
    "num_points": None,
    # (list[str] | None) columns in provided geo_raw_path to use for strata
    "strata_columns": None,
    # (list[str] | None) columns to group by for annotations generation
    "groupby_columns": None,
    # (list[str] | None) list of actions to perform on shapefile before sampling
    "preprocess_actions": None,
    # (bool) whether to flatten annotations to single dimension
    "flat_annotations": True,
    # (int | None) time combination settings
    "year_step": None,  # (int | None) number of years between each sample + 1

    # gee strata settings
    "gee_stratafied_config": {
        "num_points": None,
        "num_strata": None,
        "scale": None,
        "start_date": None,
        "end_date": None
    },

    # train test split settings
    # (float | None) ratio of validate samples from total samples
    "validate_ratio": 2e-1,
    # (float | None) ratio of test samples from validate samples
    "test_ratio": 2e-1,
    # (float | None) ratio of predict samples from test samples
    "predict_ratio": 5e-1,

    # image and downloadng settings
    # (Literal["fit", "validate", "test", "predict"]) file type to download from GEE
    "file_type": "ZARR",
    # (bool) whether to overwrite existing files
    "overwrite": False,
    # (int) scale in meters/pixel of images
    "scale": 30,
    # (int) edge size of square in meters
    "pixel_edge_size": 256,
    # (str) projection of images
    "projection": "EPSG:5070",
    # (int) n years to look back from observation date
    "look_years": 2,

    # MP and GEE specific settings
    # (int) number of parallel workers to use for annotation generation
    "num_workers": 64,
    # (int) number of parallel workers to use for download
    "gee_workers": GEE_REQUEST_LIMIT,
    # (int) number of chips to download before locking IO and writing
    "io_limit": 32,
}

if os.path.exists(SAMPLE_CONFIG_PATH):
    SAMPLER = SAMPLER | load_config(SAMPLE_CONFIG_PATH)

# pytorch lightning settings
DATALOADER = {
    "batch_size": 16,
    "num_workers": 8,
    "chip_size": SAMPLER["pixel_edge_size"],
}

CHECKPOINT = {
    "dirpath": CHECKPOINT_PATH,
    "filename": "epoch-{epoch}_val_loss-{val_loss:.2f}",
    "monitor": "val_loss",
    "save_top_k": 3,
    "auto_insert_metric_name": False,
    "every_n_epochs": 1,
}

LOGGER = {
    "name": METHOD,
    "save_dir": LOG_PATH,
    "log_graph": True,
    "default_hp_metric": False
}

if __name__ == "__main__":
    # saving sampler and download configs
    save_config(SAMPLER, SAMPLE_CONFIG_PATH)

    # generating and saving run configs for fit, validate, test and predict
    run = {
        "data": {
            "class_path": "ChipsDataModule",
            "init_args": DATALOADER
        }
    }
    for method in ["fit", "validate", "test", "predict"]:
        run_config_path = os.path.join(CONFIG_PATH, f"{method}.yaml")
        match method:
            case "fit":
                run["max_epochs"] = 256
            case "validate":
                run["verbose"] = True
            case "test":
                run["ckpt_path"] = None
                run["verbose"] = True
            case "predict":
                run["ckpt_path"] = None
                run["data"]["init_args"]["anno_data_path"] = None

        save_config(run, run_config_path)
