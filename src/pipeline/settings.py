import os
from datetime import datetime

START_DATE_STR = "1985-06-01"
END_DATE_STR = "2023-09-01"
START_DATE = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
END_DATE = datetime.strptime(END_DATE_STR, "%Y-%m-%d")

# sample information
SAMPLE_NAME = os.getenv("SUNDIAL_SAMPLE_NAME")
EXPERIMENT_SUFFIX = os.getenv("SUNDIAL_EXPERIMENT_SUFFIX")

if EXPERIMENT_SUFFIX is None or len(EXPERIMENT_SUFFIX) == 0:
    EXPERIMENT_NAME = SAMPLE_NAME
else:
    EXPERIMENT_NAME = "_".join([SAMPLE_NAME, EXPERIMENT_SUFFIX])

# base paths
BASE_PATH = os.getenv("SUNDIAL_BASE_PATH")
DATA_PATH = os.path.join(BASE_PATH, "data")
BASE_LOG_PATH = os.path.join(BASE_PATH, "logs", EXPERIMENT_NAME)
SAMPLE_PATH = os.path.join(
    DATA_PATH, "samples", EXPERIMENT_NAME)

# sample and data paths
META_DATA_PATH = os.path.join(SAMPLE_PATH, "meta_data.zarr")
STRATA_MAP_PATH = os.path.join(SAMPLE_PATH, "strata_map.yaml")
CHIP_DATA_PATH = os.path.join(SAMPLE_PATH, "chip_data.zarr")
ANNO_DATA_PATH = os.path.join(SAMPLE_PATH, "anno_data.zarr")
TRAIN_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "train_sample.zarr")
VALIDATE_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "validate_sample.zarr")
PREDICT_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "predict_sample.zarr")
TEST_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "test_sample.zarr")

# pytorch lightning paths
PREDICTIONS_PATH = os.path.join(SAMPLE_PATH, "predictions")
PYTORCH_LOG_PATH = os.path.join(BASE_LOG_PATH, "lightning")

# shapefile and source data paths
GEO_FILE_PATH = os.path.join(DATA_PATH, "shapes", SAMPLE_NAME)

# image and meta data settings
RANDOM_STATE = 42
SQUARE_COLUMNS = [f"square_{i}" for i in range(5)]
BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
NUM_CHANNELS = len(BANDS)
MASK_LABELS = ["cloud"]
STRATA_DIM_NAME = "strata"
STRATA_ATTR_NAME = "stratum"
PADDING = 1.05
GEE_REQUEST_LIMIT = 40
GEE_FEATURE_LIMIT = 1e4

# GEE file type settings
FILE_EXT_MAP = {
    "GEO_TIFF": "tif",
    "NPY": "npy",
    "NUMPY_NDARRAY": "npy",
    "ZARR": "zarr"
}

SAMPLER = {
    # sampling settings
    "generate_squares": True,
    "generate_time_combinations": True,
    "generate_train_test_split": True,
    "method": "stratified",

    # paths
    "geo_file_path": GEO_FILE_PATH,
    "meta_data_path": META_DATA_PATH,

    # strata settings
    "num_points": 1e4,  # number of points per strata
    "num_strata": 1e2,  # number of strata to generate based on stats
    "start_date": START_DATE,  # start date for medoid time samples
    "end_date": END_DATE,  # end date for medoid time samples
    "meter_edge_size": 256*30,  # edge size of square in meters
    "strata_map_path": STRATA_MAP_PATH,  # path to save strata map
    "strata_scale": 1e4,  # scale in which to split stats to generate strata
    "strata_columns": None,  # columns in provided geo_file_path to use for strata
    "fraction": None,  # fraction of total points to sample

    # train test split settings
    "back_step": 5,  # n years to step back from end date
    "validate_ratio": 2e-1,  # ratio of validate samples from total samples
    "test_ratio": 2e-1,  # ratio of test samples from validate samples
    "predict_ratio": 5e-1,  # ratio of predict samples from test samples
    "train_sample_path": TRAIN_SAMPLE_PATH,
    "validate_sample_path": VALIDATE_SAMPLE_PATH,
    "test_sample_path": TEST_SAMPLE_PATH,
    "predict_sample_path": PREDICT_SAMPLE_PATH,

    # logging and testing
    "log_path": BASE_LOG_PATH,
    "log_name": "sampler",
}

DOWNLOADER = {
    # downloading settings
    "start_date": START_DATE,  # start date for medoid time samples
    "end_date": END_DATE,  # end date for medoid time samples
    "file_type": "ZARR",  # file type to download from GEE
    "overwrite": False,  # whether to overwrite existing files
    "scale": 30,  # scale to download from GEE
    "pixel_edge_size": round(256*PADDING),  # edge size of square in pixels
    "reprojection": "UTM",  # reprojection of image (default: EPSG:4326)
    "overlap_band": False,  # whether to include overlap to og polygon band
    "back_step": SAMPLER["back_step"],  # n years to step back from end date

    # paths
    "chip_data_path": CHIP_DATA_PATH,  # path to store actual images
    "strata_map_path": STRATA_MAP_PATH,  # path to load strata map
    "anno_data_path": ANNO_DATA_PATH,  # path to load strata image
    "meta_data_path": META_DATA_PATH,  # path to store meta data of images

    # GEE specific settings
    "num_workers": 64,  # number of parallel workers to use for download and post processing
    "retries": 1,  # number of retries to use for download attempts
    "ignore_size_limit": True,  # whether to ignore the size limit for download
    "io_lock": True,  # whether to lock IO during download

    # logging and testing
    "log_path": BASE_LOG_PATH,
    "log_name": "downloader",
}

DATALOADER = {
    # name of strata column in strata_data_path
    "strata_attr_name": STRATA_ATTR_NAME,
    "file_type": FILE_EXT_MAP[DOWNLOADER["file_type"]],
    "chip_data_path": CHIP_DATA_PATH,
    "anno_data_path": ANNO_DATA_PATH,
    "train_sample_path": TRAIN_SAMPLE_PATH,
    "validate_sample_path": VALIDATE_SAMPLE_PATH,
    "test_sample_path": TEST_SAMPLE_PATH,
    "predict_sample_path": PREDICT_SAMPLE_PATH,
    "chip_size": round(SAMPLER["meter_edge_size"] / DOWNLOADER["scale"]),
    "base_year": END_DATE.year,
    "back_step": SAMPLER["back_step"]
}

PY_WRITER = {
    "output_dir": PREDICTIONS_PATH,
    "write_interval": "batch"
}

TENSORBOARD_LOGGER = {
    "root_dir": PYTORCH_LOG_PATH,
    "name": EXPERIMENT_NAME
}
