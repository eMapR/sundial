import os
from datetime import datetime

START_DATE_STR = "1985-06-01"
END_DATE_STR = "2023-09-01"
START_DATE = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
END_DATE = datetime.strptime(END_DATE_STR, "%Y-%m-%d")

BASE_PATH = os.getenv("SUNDIAL_BASE_PATH")
DATA_PATH = os.path.join(BASE_PATH, "data")
BASE_LOG_PATH = os.path.join(BASE_PATH, "logs")

SAMPLE_PATH = os.path.join(
    DATA_PATH, "samples", os.getenv("SUNDIAL_SAMPLE_NAME"))
META_DATA_PATH = os.path.join(SAMPLE_PATH, "meta_data.zarr")
CHIP_DATA_PATH = os.path.join(SAMPLE_PATH, "chip_data.zarr")
TRAIN_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "train_sample.zarr")
VALIDATE_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "validate_sample.zarr")
PREDICT_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "predict_sample.zarr")
TEST_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "test_sample.zarr")

SQUARE_COLUMNS = [f"square_{i}" for i in range(5)]
BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
MASK_LABELS = ["cloud"]

CHIP_SIZE = 253  # Google will add it's own padding to get to 256
SCALE = 30
PADDING = 1.05

BACK_STEP = 5
RANDOM_STATE = 42
NUM_CHANNELS = len(BANDS)

GEE_FILE_TYPE = "ZARR"
FILE_EXT_MAP = {
    "GEO_TIFF": "tif",
    "NPY": "npy",
    "NUMPY_NDARRAY": "npy",
    "ZARR": "zarr"
}

SAMPLER = {
    "generate_squares": True,
    "generate_time_samples": True,
    "method": "stratified",
    "geo_file_path": os.path.join(DATA_PATH, "shapes", os.getenv("SUNDIAL_SAMPLE_NAME")),
    "num_points": 1e4,
    "num_strata": 1e2,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "edge_size": round(CHIP_SIZE * SCALE),
    "strata_scale": 1e4,
    "strata_columns": None,
    "fraction": None,
    "back_step": BACK_STEP,
    "training_ratio": 2e-1,
    "test_ratio": 2e-3,
    "meta_data_path": META_DATA_PATH,
    "train_sample_path": TRAIN_SAMPLE_PATH,
    "validate_sample_path": VALIDATE_SAMPLE_PATH,
    "test_sample_path": TEST_SAMPLE_PATH,
    "overwrite": True,
    "log_path": BASE_LOG_PATH,
    "log_name": "sundial.sampler",
    "test": False,
}

DOWNLOADER = {
    "start_date": START_DATE,
    "end_date": END_DATE,
    "file_type": GEE_FILE_TYPE,
    "scale": SCALE,
    "edge_size": round((SAMPLER["edge_size"]/SCALE)*PADDING),
    "reprojection": "UTM",
    "overlap_band": False,
    "chip_data_path": CHIP_DATA_PATH,
    "meta_data_path": META_DATA_PATH,
    "num_workers": 64,
    "retries": 1,
    "request_limit": 40,
    "ignore_size_limit": True,
    "io_lock": True,
    "overwrite": False,
    "log_path": BASE_LOG_PATH,
    "log_name": "sundial.downloader",
    "test": False,
}

DATALOADER = {
    "file_type": FILE_EXT_MAP[GEE_FILE_TYPE],
    "chip_data_path": CHIP_DATA_PATH,
    "train_sample_path": TRAIN_SAMPLE_PATH,
    "validate_sample_path": VALIDATE_SAMPLE_PATH,
    "test_sample_path": TEST_SAMPLE_PATH,
    "predict_sample_path": PREDICT_SAMPLE_PATH,
    "chip_size": CHIP_SIZE+3,
    "base_year": END_DATE.year,
    "back_step": BACK_STEP
}
