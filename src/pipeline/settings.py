import os
import yaml

from datetime import date

# config save and load functions


def save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# max date ranges as default
START_DATE = date(1985, 6, 1)
END_DATE = date(1985, 9, 1)

# sample information
SAMPLE_NAME = os.getenv("SUNDIAL_SAMPLE_NAME")
EXPERIMENT_SUFFIX = os.getenv("SUNDIAL_EXPERIMENT_SUFFIX")
EXPERIMENT_NAME = os.getenv("SUNDIAL_EXPERIMENT_NAME")

# base paths
BASE_PATH = os.getenv("SUNDIAL_BASE_PATH")
CONFIGS_PATH = os.path.join(BASE_PATH, "configs")
SHAPES_PATH = os.path.join(BASE_PATH, "shapes")
SAMPLES_PATH = os.path.join(BASE_PATH, "samples")
CHECKPOINTS_PATH = os.path.join(BASE_PATH, "checkpoints")
LOGS_PATH = os.path.join(BASE_PATH, "logs")

# experiment paths
CONFIG_PATH = os.path.join(CONFIGS_PATH, EXPERIMENT_NAME)
SAMPLE_PATH = os.path.join(SAMPLES_PATH, EXPERIMENT_NAME)
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_PATH, EXPERIMENT_NAME)
LOG_PATH = os.path.join(LOGS_PATH, EXPERIMENT_NAME)

# config paths
SAMPLE_CONFIG_PATH = os.path.join(CONFIG_PATH, "sample.yaml")
DOWNLOAD_CONFIG_PATH = os.path.join(CONFIG_PATH, "download.yaml")

# sample and data paths
META_DATA_PATH = os.path.join(SAMPLE_PATH, "meta_data.zarr")
STRATA_MAP_PATH = os.path.join(SAMPLE_PATH, "strata_map.yaml")
CHIP_DATA_PATH = os.path.join(SAMPLE_PATH, "chip_data.zarr")
ANNO_DATA_PATH = os.path.join(SAMPLE_PATH, "anno_data.zarr")
TRAIN_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "train_sample.npy")
VALIDATE_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "validate_sample.npy")
PREDICT_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "predict_sample.npy")
TEST_SAMPLE_PATH = os.path.join(SAMPLE_PATH, "test_sample.npy")


# shapefile and source data paths
GEO_FILE_PATH = os.path.join(SHAPES_PATH, SAMPLE_NAME)

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
NO_DATA_VALUE = -9999

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
    "log_path": LOG_PATH,
    "log_name": "sample",
}
if os.path.exists(SAMPLE_CONFIG_PATH):
    SAMPLER = SAMPLER | load_config(SAMPLE_CONFIG_PATH)

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
    "anno_data_path": ANNO_DATA_PATH,  # path to load strata image
    "strata_map_path": STRATA_MAP_PATH,  # path to load strata map
    "meta_data_path": META_DATA_PATH,  # path to store meta data of images

    # MP and GEE specific settings
    "num_workers": GEE_REQUEST_LIMIT,  # number of parallel workers to use for download
    "io_limit": 32,  # number of chips to download before locking IO and writing

    # logging and testing
    "log_path": LOG_PATH,
    "log_name": "download",
}
if os.path.exists(DOWNLOAD_CONFIG_PATH):
    DOWNLOADER = DOWNLOADER | load_config(DOWNLOAD_CONFIG_PATH)


# pytorch lightning settings
DATALOADER = {
    # image / chip settings
    "chip_size": round(SAMPLER["meter_edge_size"] / DOWNLOADER["scale"]),
    "base_year": START_DATE.year,
    "back_step": SAMPLER["back_step"],

    # path settings to load data
    "file_type": FILE_EXT_MAP[DOWNLOADER["file_type"]],
    "chip_data_path": CHIP_DATA_PATH,
    "anno_data_path": ANNO_DATA_PATH,
    "train_sample_path": TRAIN_SAMPLE_PATH,
    "validate_sample_path": VALIDATE_SAMPLE_PATH,
    "test_sample_path": TEST_SAMPLE_PATH,
    "predict_sample_path": PREDICT_SAMPLE_PATH,
}

CHECKPOINT = {
    "dirpath": CHECKPOINT_PATH,
    "filename": "epoch-{epoch:02d}_val_loss-{val_loss:.2f}",
    "monitor": "val_loss",
    "save_top_k": 4,
    "auto_insert_metric_name": False,
    "every_n_epochs": 2,
}

LOGGER = {
    "save_dir": LOG_PATH
}

if __name__ == "__main__":
    # saving sampler and download configs
    save_config(SAMPLER, SAMPLE_CONFIG_PATH)
    save_config(DOWNLOADER, DOWNLOAD_CONFIG_PATH)

    # generating and saving run configs for fit, validate, test and predict
    run = {
        "data": {
            "class_path": "ChipsDataModule",
            "init_args": DATALOADER
        },
        
        "trainer": {
            "accelerator": "cuda",
            "callbacks": [{"class_path": "SundialPrithviCallback"}],
            "logger": {
                "class_path": "TensorBoardLogger",
                "init_args": LOGGER
            },
            "profiler": "advanced",
        }
    }
    for method in ["fit", "validate", "test", "predict"]:
        run_config_path = os.path.join(CONFIG_PATH, f"{method}.yaml")
        run["trainer"]["logger"]["init_args"]["name"] = method
        match method:
            case "fit":
                run["trainer"]["inference_mode"] = True
                run["trainer"]["callbacks"].append({
                    "class_path": "ModelCheckpoint",
                    "init_args": CHECKPOINT
                })
            case "validate" | "predict" | "test":
                run["trainer"]["inference_mode"] = False

        save_config(run, run_config_path)
