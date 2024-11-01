import os
import yaml


# convenience functions for saving and loading yaml files
def save_yaml(config: dict, path: str | os.PathLike):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        return config if config else {}

def update_yaml(config: dict, path: str | os.PathLike) -> dict:
    if os.path.exists(path):
        old_config = load_yaml(path)
        config = old_config | config
    save_yaml(config, path)


# experiment information
SAMPLE_NAME = os.getenv("SUNDIAL_SAMPLE_NAME")
EXPERIMENT_PREFIX = os.getenv("SUNDIAL_EXPERIMENT_PREFIX")
EXPERIMENT_SUFFIX = os.getenv("SUNDIAL_EXPERIMENT_SUFFIX")
EXPERIMENT_BASE_NAME = os.getenv("SUNDIAL_EXPERIMENT_BASE_NAME")
EXPERIMENT_FULL_NAME = os.getenv("SUNDIAL_EXPERIMENT_FULL_NAME")
METHOD = os.getenv("SUNDIAL_METHOD")
JOB_NAME = f"{METHOD}_{EXPERIMENT_FULL_NAME}"

# base paths
BASE_PATH = os.getenv("SUNDIAL_BASE_PATH")
CONFIGS_PATH = os.path.join(BASE_PATH, "configs")
SHAPES_PATH = os.path.join(BASE_PATH, "shapes")
SAMPLES_PATH = os.path.join(BASE_PATH, "samples")
CHECKPOINTS_PATH = os.path.join(BASE_PATH, "checkpoints")
PREDICTIONS_PATH = os.path.join(BASE_PATH, "predictions")
LOGS_PATH = os.path.join(BASE_PATH, "logs")

# experiment paths
CONFIG_PATH = os.path.join(CONFIGS_PATH, EXPERIMENT_BASE_NAME)
SAMPLE_PATH = os.path.join(SAMPLES_PATH, EXPERIMENT_BASE_NAME)
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_PATH, EXPERIMENT_BASE_NAME)
PREDICTION_PATH = os.path.join(PREDICTIONS_PATH, EXPERIMENT_BASE_NAME)
LOG_PATH = os.path.join(LOGS_PATH, EXPERIMENT_BASE_NAME)

# config paths
SAMPLE_CONFIG_PATH = os.path.join(CONFIG_PATH, "sample.yaml")
METHOD_CONFIG_PATH = os.path.join(CONFIG_PATH, f"{METHOD}.yaml")
BASE_CONFIG_PATH = os.path.join(CONFIG_PATH, "base.yaml")

# sample and data paths
META_DATA_PATH = os.path.join(SAMPLE_PATH, "meta_data")
STAT_DATA_PATH = os.path.join(SAMPLE_PATH, "stat_data.yaml")
CHIP_DATA_PATH = os.path.join(SAMPLE_PATH, "chip_data")
ANNO_DATA_PATH = os.path.join(SAMPLE_PATH, "anno_data")
ALL_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "all_sample.npy")
TRAIN_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "train_sample.npy")
VALIDATE_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "validate_sample.npy")
PREDICT_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "predict_sample.npy")
TEST_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "test_sample.npy")

# shapefile and source data paths
GEO_RAW_PATH = os.path.join(SHAPES_PATH, SAMPLE_NAME)
GEO_POP_PATH = os.path.join(SAMPLE_PATH, "gpop_data")

# non configurable GEE, image, and meta data settings
RANDOM_SEED = 42
STRATA_LABEL = "strata"
DATETIME_LABEL = "datetime"
NO_DATA_VALUE = 0
GEE_REQUEST_LIMIT = 40
EE_END_POINT = 'https://earthengine-highvolume.googleapis.com'

# GEE file type mapping to file extension
FILE_EXT_MAP = {
    "GEO_TIFF": "tif",
    "NPY": "npy",
    "NUMPY_NDARRAY": "npy",
    "ZARR": "zarr"
}

# configs relating to sampler methods
SAMPLER_CONFIG = {
    # date information for medoid composites
    # (dict | None) image generator / parser kwargs for download
    "parser_kwargs": {
        "start_month": 7,
        "start_day": 15,
        "end_month": 9,
        "end_day": 1
    },

    # sampling settings
    # (str) method to use for sampling
    "method": "centroid",
    # (float| int | None) number of points to sample. If float, a fraction of sample = n is used.
    "num_points": 2.0e-2,
    # (List[str] | None) columns in provided geo_raw_path to use for strata
    "strata_columns": None,
    # (List[str] | None) columns to group by for annotation generation in addition to strata columns
    "groupby_columns": None,
    # (dict) List of actions to perform on shapefile before sampling
    "preprocess_actions": [],
    # (dict) List of actions to perform on chip and anno data after sampling
    "postprocess_actions": [],
    # (str) Column to use for datetime value in geo file
    "datetime_column": "year",
    # (bool) Toggle for splitting time range at one location into multiple samples
    "generate_time_combinations": False,
    # (int | None) number of time steps between each sample, or number of years to include in model including observation year
    "time_step": None,

    # square generation settings
    # (dict | None) settings for passing to square generator function
    "squares_config": {},

    # Tuple(float) | None ratio of validate, test samples from total samples
    "split_ratios": [2e-1, 2e-2],

    # image and downloadng settings
    # (Literal["GEO_TIFF", "ZARR", "NPY", "NUMPY_NDARRAY"]) file type to download from GEE
    "file_type": "ZARR",
    # (bool) whether to overwrite existing files
    "overwrite": False,
    # (int) scale in meters/pixel of images
    "scale": 30,
    # (int) edge size of square in meters
    "pixel_edge_size": 256,
    # (str) projection to save polygons and images in
    "projection": "EPSG:5070",
    # (int) n time step to look back from observation date (i.e. 2 = 3 years total including observation year)
    "look_range": 2,
    # (str) function in pipeline/utils to parse metadata
    "meta_data_parser": "medoid_from_year",
    # (str) function in pipeline/utils to generate expression in google earth engine
    "ee_image_factory": "lt_medoid_image_factory",
    # (str) function in pipeline/utils to reshape resulting download from GEE
    "image_reshaper": "unstack_band_years",

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
    SAMPLER_CONFIG |= load_yaml(SAMPLE_CONFIG_PATH)

if SAMPLER_CONFIG["file_type"] == "ZARR":
    ext = FILE_EXT_MAP[SAMPLER_CONFIG["file_type"]]
    CHIP_DATA_PATH = CHIP_DATA_PATH + f".{ext}"
    ANNO_DATA_PATH = ANNO_DATA_PATH + f".{ext}"
