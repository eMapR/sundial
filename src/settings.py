import os
from datetime import datetime
import torch.nn as nn

START_DATE_STR = "1985-06-01"
END_DATE_STR = "2023-09-01"
START_DATE = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
END_DATE = datetime.strptime(END_DATE_STR, "%Y-%m-%d")

BASE_PATH = os.getenv("SUNDIAL_BASE_PATH")
META_DATA_PATH = os.path.join(BASE_PATH, "meta_data.zarr")
CHIP_DATA_PATH = os.path.join(BASE_PATH, "chip_data.zarr")
TRAINING_DATA_PATH = os.path.join(BASE_PATH, "training_data.zarr")
VALIDATE_DATA_PATH = os.path.join(BASE_PATH, "validate_data.zarr")
PREDICT_DATA_PATH = os.path.join(BASE_PATH, "predict_data.zarr")
BASE_LOG_PATH = os.path.join(BASE_PATH, "logs")

SQUARE_COLUMNS = [f"square_{i}" for i in range(5)]
BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
MASK_LABELS = ["cloud"]

CHIP_SIZE = 256
BACK_STEP = 5

SAMPLER = {
    "start_year": START_DATE.year,
    "end_year": END_DATE.year,
    "method": "gaussian",
    "num_points": 100,
    "std_dev": 10,
    "edge_size": .7,
    "num_squares": 100,
    "meta_data_path": META_DATA_PATH,
    "training_data_path": TRAINING_DATA_PATH,
    "validate_data_path": VALIDATE_DATA_PATH,
    "predict_data_path": PREDICT_DATA_PATH,
    "log_path": BASE_LOG_PATH,
    "log_name": "sundial.sampler.logs"
}

DOWNLOADER = {
    "start_date": START_DATE,
    "end_date": END_DATE,
    "file_type": "ZARR",
    "out_path": CHIP_DATA_PATH,
    "meta_data_path": META_DATA_PATH,
    "num_workers": 64,
    "retries": 3,
    "request_limit": 40,
    "overwrite": False,
    "verbose": False,
    "log_path": BASE_LOG_PATH,
    "log_name": "sundial.downloader.logs"
}

DATAMODULE = {
    "chips_path": CHIP_DATA_PATH,
    "training_samples_path": TRAINING_DATA_PATH,
    "validate_samples_path": VALIDATE_DATA_PATH,
    "predict_samples_path": PREDICT_DATA_PATH,
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
