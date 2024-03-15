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


# max date ranges as default
START_DATE = date(1985, 6, 1)
END_DATE = date(2023, 9, 1)

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
    "preprocess_data": True,
    "stratified_sample": True,
    "generate_squares": True,
    "generate_time_combinations": True,
    "generate_train_test_splits": True,
    "generate_annotation_data": True,
    "generate_image_chip_data": True,

    # sampling settings
    "method": "centroid",
    "fraction": 2.0e-2,  # fraction of total points to sample to pull from
    "num_points": None,  # number of points to sample. This is used if fraction is None
    "strata_columns": None,  # columns in provided geo_raw_path to use for strata
    "groupby_columns": None,  # columns to group by for annotations generation
    "preprocess_actions": None,  # list of actions to perform on shapefile before sampling
    "flat_annotations": True,

    # time_combinations settings
    "back_step": None,  # number of years to step back from end date

    # gee strata settings
    "gee_num_points": 1e4,  # number of points per strata
    "gee_num_strata": 1e2,  # number of strata to generate based on stats
    "gee_strata_scale": 1e4,  # scale in which to split stats to generate strata
    "gee_start_date": START_DATE,  # start date for medoid time samples
    "gee_end_date": END_DATE,  # end date for medoid time samples

    # train test split settings
    "validate_ratio": 2e-1,  # ratio of validate samples from total samples
    "test_ratio": 2e-1,  # ratio of test samples from validate samples
    "predict_ratio": 5e-1,  # ratio of predict samples from test samples

    # image and downloading settings
    "file_type": "ZARR",  # file type to download from GEE
    "overwrite": False,  # whether to overwrite existing files
    "scale": 30,  # scale in meters/pixel of images
    "pixel_edge_size": 256,  # edge size of square in meters
    "projection": "EPSG:5070",  # projection of images
    "look_years": 3,  # n years to step back from end date

    # MP and GEE specific settings
    "num_workers": 64,  # number of parallel workers to use for annotation generation
    "gee_workers": GEE_REQUEST_LIMIT,  # number of parallel workers to use for download
    "io_limit": 32,  # number of chips to download before locking IO and writing
}

if os.path.exists(SAMPLE_CONFIG_PATH):
    SAMPLER = SAMPLER | load_config(SAMPLE_CONFIG_PATH)

# pytorch lightning settings
DATALOADER = {
    # image / chip settings
    "chip_size": SAMPLER["pixel_edge_size"],
    "base_year": START_DATE.year,
    "back_step": SAMPLER["back_step"],
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
        },
        "trainer": {
            "log_every_n_steps": 16,
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
