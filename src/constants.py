import os

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
DATA_PATH = os.path.join(BASE_PATH, "data")
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
APPEND_DIM = "sample"
CLASS_LABEL = "class"
DATETIME_LABEL = "datetime"
NO_DATA_VALUE = 0
IDX_NAME_ZFILL = 8
GEE_REQUEST_LIMIT = 40
EE_END_POINT = 'https://earthengine-highvolume.googleapis.com'

# GEE file type mapping to file extension
FILE_EXT_MAP = {
    "GEO_TIFF": "tif",
    "NPY": "npy",
    "NUMPY_NDARRAY": "npy",
    "ZARR": "zarr"
}
