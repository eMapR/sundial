import os
from datetime import datetime

START_DATE_STR = "1985-06-01"
END_DATE_STR = "2023-09-01"
START_DATE = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
END_DATE = datetime.strptime(END_DATE_STR, "%Y-%m-%d")

BASE_PATH = os.getenv("SUNDIAL_BASE_PATH")
DATA_PATH = os.path.join(BASE_PATH, "data")
SAMPLE_PATH = os.path.join(
    DATA_PATH, "samples", os.getenv("SUNDIAL_SAMPLE_NAME"))

META_DATA_PATH = os.path.join(SAMPLE_PATH, "meta_data.zarr")
CHIP_DATA_PATH = os.path.join(SAMPLE_PATH, "chip_data.zarr")
TRAINING_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "training_sample.zarr")
VALIDATE_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "validate_sample.zarr")
PREDICT_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "predict_sample.zarr")
TEST_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "test_sample.zarr")
BASE_LOG_PATH = os.path.join(DATA_PATH, "logs")

SQUARE_COLUMNS = [f"square_{i}" for i in range(5)]
BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
MASK_LABELS = ["cloud"]

CHIP_SIZE = 253  # Google will add it's own padding to get to 256
SCALE = 30
PADDING = 1.05

BACK_STEP = 5
RANDOM_STATE = 42

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
    "projection": None,
    "fraction": None,
    "back_step": BACK_STEP,
    "training_ratio": 2e-1,
    "test_ratio": 2e-3,
    "meta_data_path": META_DATA_PATH,
    "training_sample_path": TRAINING_SAMPLE_PATH,
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
    "file_type": "ZARR",
    "scale": SCALE,
    "edge_size": round((SAMPLER["edge_size"]/SCALE)*PADDING),
    "reproject": "UTM",
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

DATAMODULE = {
    "chip_data_path": CHIP_DATA_PATH,
    "training_sample_path": TRAINING_SAMPLE_PATH,
    "validate_sample_path": PREDICT_SAMPLE_PATH,
    "test_sample_path": TEST_SAMPLE_PATH,
    "predict_sample_path": PREDICT_SAMPLE_PATH,
    "batch_size": 1024,
    "num_workers": 16,
    "chip_size": CHIP_SIZE,
    "base_year": END_DATE.year,
    "back_step": BACK_STEP
}

SUNDIAL = {
    "image_size": 256,
    "patch_size": 16,
    "num_channels": 6,
    "num_frames": BACK_STEP + 1,
    "tubelet_size": 1,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "qkv_bias": True,
    "use_mean_pooling": True,
    "decoder_num_attention_heads": 6,
    "decoder_hidden_size": 384,
    "decoder_num_hidden_layers": 4,
    "decoder_intermediate_size": 1536,
    "norm_pix_loss": True,
    "learning_rate": 1e-3,
}
