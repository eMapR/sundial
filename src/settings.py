import os
from datetime import datetime

START_DATE_STR = "1985-06-01"
END_DATE_STR = "2023-09-01"
START_DATE = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
END_DATE = datetime.strptime(END_DATE_STR, "%Y-%m-%d")

BASE_PATH = os.getenv("SUNDIAL_BASE_PATH")
META_DATA_PATH = os.path.join(BASE_PATH, "meta_data.zarr")
CHIP_DATA_PATH = os.path.join(BASE_PATH, "chip_data.zarr")
TRAINING_SAMPLES_PATH = os.path.join(BASE_PATH, "training_samples.zarr")
VALIDATE_SAMPLES_PATH = os.path.join(BASE_PATH, "validate_samples.zarr")
PREDICT_SAMPLES_PATH = os.path.join(BASE_PATH, "predict_samples.zarr")
TEST_SAMPLES_PATH = os.path.join(BASE_PATH, "test_samples.zarr")
BASE_LOG_PATH = os.path.join(BASE_PATH, "logs")

SQUARE_COLUMNS = [f"square_{i}" for i in range(5)]
BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
MASK_LABELS = ["cloud"]

CHIP_SIZE = 256
SCALE = 30
BACK_STEP = 5

SAMPLER = {
    "start_date": START_DATE,
    "end_date": END_DATE,
    "method": "stratified",
    "file_path": os.path.join(BASE_PATH, "blm_or_wa_bounds"),
    "num_points": 1e4,
    "num_strata": 1e2,
    "strat_scale": 1e4,
    "edge_size": 7.7e3,
    "back_step": BACK_STEP,
    "meta_data_path": META_DATA_PATH,
    "training_samples_path": TRAINING_SAMPLES_PATH,
    "validate_samples_path": PREDICT_SAMPLES_PATH,
    "test_samples_path": TEST_SAMPLES_PATH,
    "log_path": BASE_LOG_PATH,
    "log_name": "sundial.sampler",
    "test": False,
}

DOWNLOADER = {
    "start_date": START_DATE,
    "end_date": END_DATE,
    "file_type": "ZARR",
    "scale": SCALE,
    "edge_size": round((SAMPLER["edge_size"]/SCALE)*1.05),
    "reproject": "UTM",
    "chip_data_path": CHIP_DATA_PATH,
    "meta_data_path": META_DATA_PATH,
    "num_workers": 64,
    "retries": 1,
    "request_limit": 40,
    "ignore_size_limit": True,
    "overwrite": False,
    "log_path": BASE_LOG_PATH,
    "log_name": "sundial.downloader",
    "test": False,
}

DATAMODULE = {
    "chips_path": CHIP_DATA_PATH,
    "training_samples_path": TRAINING_SAMPLES_PATH,
    "validate_samples_path": PREDICT_SAMPLES_PATH,
    "test_samples_path": TEST_SAMPLES_PATH,
    "predict_samples_path": PREDICT_SAMPLES_PATH,
    "batch_size": 1024,
    "num_workers": 16,
    "chip_size": CHIP_SIZE,
    "base_year": END_DATE.year,
    "back_step": BACK_STEP
}

SUNDIAL = {
    "num_frames": BACK_STEP + 1,
    "num_channels": len(BANDS),
}
