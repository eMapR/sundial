import os
import yaml


def save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def update_config(config, path):
    if os.path.exists(path):
        old_config = load_config(path)
        config |= old_config
    save_config(config, path)


# sample information
SAMPLE_NAME = os.getenv("SUNDIAL_SAMPLE_NAME")
EXPERIMENT_PREFIX = os.getenv("SUNDIAL_EXPERIMENT_PREFIX")
EXPERIMENT_SUFFIX = os.getenv("SUNDIAL_EXPERIMENT_SUFFIX")
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
CHIP_DATA_PATH = os.path.join(SAMPLE_PATH, "chip_data")
ANNO_DATA_PATH = os.path.join(SAMPLE_PATH, "anno_data")
TRAIN_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "train_sample.npy")
VALIDATE_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "validate_sample.npy")
PREDICT_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "predict_sample.npy")
TEST_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "test_sample.npy")

# zipped shapefile and source data paths
GEO_RAW_PATH = os.path.join(SHAPES_PATH, SAMPLE_NAME)
GEO_PRE_PATH = os.path.join(SAMPLE_PATH, "geo_file")

# non configurable GEE, image, and meta data settings
RANDOM_STATE = 42
MASK_LABELS = ["cloud"]
STRATA_ATTR_NAME = "stratum"
NO_DATA_VALUE = 0
GEE_REQUEST_LIMIT = 40
EE_END_POINT = 'https://earthengine-highvolume.googleapis.com'

# GEE file type settings
FILE_EXT_MAP = {
    "GEO_TIFF": "tif",
    "NPY": "npy",
    "NUMPY_NDARRAY": "npy",
    "ZARR": "zarr"
}

SAMPLER_CONFIG = {
    # sampling settings
    # (dict) toggles for subprocesses within the sampler
    "sample_toggles": {
        "preprocess_data": True,
        "stratify_data": True,
        "generate_squares": True,
        "generate_time_combinations": False,
        "generate_train_test_splits": True,
    },

    # date information for medoid composites
    # (dict | None) image generator / parser kwargs for download
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
    # (dict | None) settings for strata generation (None for no strata generation
    "gee_stratafied_config": {
        # (int) number of points to generate per stratum
        "num_points": None,
        # (int) number of strata to generate per data source
        "num_strata": None,
        # (int) scale to perform stratification
        "scale": None,
        # (int) start date to filter source images
        "start_date": None,
        # (int) end date to filter source images
        "end_date": None,
        # (list[Literal["prism", "elevation", "ads"]]) data sources to use for stratification
        "sources": None
    },

    # train test split settings
    # (float | None) ratio of validate samples from total samples
    "validate_ratio": 2e-1,
    # (float | None) ratio of test samples from validate samples
    "test_ratio": 2e-1,
    # (float | None) ratio of predict samples from test samples
    "predict_ratio": 5e-1,

    # image and downloadng settings
    # (Literal["GEO_TIFF", "ZARR", "NPY", "NUMPY_NDARRAY"]) file type to download from GEE
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

# loading sampler config if it exists
if os.path.exists(SAMPLE_CONFIG_PATH):
    SAMPLER_CONFIG = SAMPLER_CONFIG | load_config(SAMPLE_CONFIG_PATH)

if SAMPLER_CONFIG["file_type"] == "ZARR":
    ext = FILE_EXT_MAP[SAMPLER_CONFIG["file_type"]]
    CHIP_DATA_PATH = CHIP_DATA_PATH + f".{ext}"
    ANNO_DATA_PATH = ANNO_DATA_PATH + f".{ext}"

# default lightning dataloader settings
DATALOADER_CONFIG = {
    "batch_size": 32,
    "num_workers": 16,
    "chip_size": SAMPLER_CONFIG["pixel_edge_size"],
}

# default lightning model checkpoint save settings
CHECKPOINT_CONFIG = {
    "dirpath": CHECKPOINT_PATH,
    "filename": "epoch-{epoch}_val_loss-{val_loss:.2f}",
    "monitor": "val_loss",
    "save_top_k": 8,
    "auto_insert_metric_name": False,
    "save_weights_only": False,
    "every_n_epochs": 1,
    "enable_version_counter": True
}

# default lightning logger settings
LOGGER_CONFIG = {
    "api_key": os.getenv("COMET_API_KEY"),
    "workspace": os.getenv("COMET_WORKSPACE"),
    "save_dir": LOG_PATH,
    "project_name": SAMPLE_NAME.replace("_", "-"),
    "experiment_name": f"{os.getenv('SUNDIAL_METHOD')}_{EXPERIMENT_NAME}",
    "log_code": False,
    "auto_param_logging": False,
    "auto_metric_logging": False,
    "auto_metric_step_rate": 16,
    "log_git_metadata": False,
    "log_git_patch": False,
}
if EXPERIMENT_SUFFIX:
    LOGGER_CONFIG["experiment_name"] += f"_{EXPERIMENT_SUFFIX}"
    CHECKPOINT_CONFIG["filename"] = f"{EXPERIMENT_SUFFIX}_" + CHECKPOINT_CONFIG["filename"]

if __name__ == "__main__":
    # saving sampler and download configs
    save_config(SAMPLER_CONFIG, SAMPLE_CONFIG_PATH)

    # creating run configs for base, fit, validate, test, and predict
    for method in ["base", "fit", "validate", "test", "predict"]:
        config_path = os.path.join(CONFIG_PATH, f"{method}.yaml")
        match method:
            case "base":
                run_config = {
                    "model": None,
                    "data": {
                        "class_path": "ChipsDataModule",
                        "init_args": DATALOADER_CONFIG
                    }
                }
            case "fit":
                run_config = {
                    "max_epochs": 256,
                }
            case "validate":
                run_config = {
                    "verbose": True,
                }
            case "test":
                run_config = {
                    "ckpt_path": None,
                    "verbose": True,
                }
            case "predict":
                run_config = {
                    "ckpt_path": None,
                    "data": {
                        "init_args": {
                            "anno_data_path": None
                        }
                    }
                }

        save_config(run_config, config_path)
