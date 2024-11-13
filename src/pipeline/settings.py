import os

from pipeline.utils import  load_yaml, recursive_merge


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
CLASS_LABEL = "strata"
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

# configs relating to sampler methods
SAMPLER_CONFIG = {
    # Sampling settings

    # (str) Method to used for generating squares around the polygons given in the original geodataframe.
    # See pipeline.generate_squares.
    "method": "centroid",

    # (float| int | None) Number of points to sample. If float, a fraction of sample = n is used. If None, all samples are included.
    # See pipeline.stratified_sample.
    "num_points": 2.0e-2,
    
    # (List[str] | None) Columns in provided in file at GEO_RAW_PATH to use for defining classes.
    # See pipeline.preprocess_data.
    "class_columns": None,

    # (List[str] | None) Columns in grouping during annotation generation in addition to class columns.
    # See pipeline.annotator.
    "groupby_columns": None,

    # (dict) List of actions to perform on shapefile before sampling.
    # See pipeline.preprocess_actions.
    "preprocess_actions": [],

    # (dict) List of actions to perform on chip and anno data after sampling.
    # See pipeline.postprocess_actions.
    "postprocess_actions": [],

    # (str) Column to use for saving datetime value in file at GEO_POP_PATH. This is currently unused.
    # See pipeline.preprocess_actions.
    "datetime_column": "year",

    # (bool) Toggle for generating sliding time windows into sample as separate samples to be injested into the model. This is currently unused.
    # See pipeline.generate_time_combinations.
    "generate_time_combinations": False,

    # (int | None) Number of time steps between each sample. This is currently unused.
    # See pipeline.generate_time_combinations.
    "time_step": None,

    # Square generation settings

    # (dict | None) settings for passing to square generator function.
    # See pipeline.generate_squares
    "squares_config": {},

    # Tuple(float) | None ratio of [validate, test] samples from total samples.
    # See utils.train_validate_test_split.
    "split_ratios": [2e-1, 2e-2],

    # Image and downloadng settings

    # (Literal["GEO_TIFF", "ZARR", "NPY", "NUMPY_NDARRAY"]) file type to download from GEE.
    # See downloader.Downloader.image_consumer.
    "file_type": "ZARR",
    
    # (bool) Whether to overwrite existing files.
    # See downloader.Downloader.image_generator.
    "overwrite": False,
    
    # (int) Scale in meters/pixel of images.
    # See downloader.Downloader.image_generator.
    "scale": 30,
    
    # (int) Edge size of square in meters.
    # See downloader.Downloader.image_generator.
    "pixel_edge_size": 256,
    
    # (str) Projection to save polygons and images. Will reproject coordinates if necessary.
    # See downloader.Downloader.image_generator.
    "projection": "EPSG:5070",
    
    # (int) Number of time steps to look back from observation date (i.e. 2 = 3 years total including observation year). Currently only years is supported.
    # See downloader.Downloader.image_generator.
    "look_range": 3,
    
    # (str) Name of function in pipeline.meta_data_parser to parse metadata in Downloader. An example is provided but more can be defined there.
    # Must consume (META_DATA_PATH, index: int, **kwargs).
    "meta_data_parser": "medoid_from_year",
    
    # (dict) Kwargs to be passed to meta_data_parser.
    "parser_kwargs": {
        "start_month": 7,
        "start_day": 15,
        "end_month": 9,
        "end_day": 1,
        "look_range": 3
    },
    
    # (str) Name of function in pipeline.ee_image_factory to generate expression in google earth engine consumed by Downloader. An example is provided but more can be defined there.
    # Must consume (square_coords: list[tuple[float, float]], start_date: datetime, end_date: datetime, pixel_edge_size: int, scale: int, projection: str, **kwargs)
    "ee_image_factory": "lt_medoid_image_factory",
    
    # (dict) Kwargs to be passed to ee_image_factory.
    "factory_kwargs": {},
    
    # (str) Name of function in pipeline.image_reshaper to reshape resulting download from GEE via Downloader. An example is provided but more can be defined there.
    # Must consume (arr: np.ndarray, index_name: str, pixel_edge_size: int,square_name: str,point_name: str, attributes: Optional[dict] = {}, **kwargs)
    "image_reshaper": "unstack_band_years",
    
    # (dict) Kwargs to be passed to image_reshaper.
    "reshaper_kwargs": {},
    
    # MP and GEE specific settings

    # (int) Number of parallel workers to use for annotation generation.
    # See pipeline.annotator.
    "num_workers": 64,
    
    # (int) Number of parallel workers to use for download.
    # See downloader.Downloader.
    "gee_workers": GEE_REQUEST_LIMIT,
    
    # (int) Number of chips to download before locking IO and writing.
    # See downloader.Downloader._image_consumer.
    "io_limit": 32,
}

# loading sampler config if it exists
if os.path.exists(SAMPLE_CONFIG_PATH):
    SAMPLER_CONFIG = recursive_merge(SAMPLER_CONFIG, load_yaml(SAMPLE_CONFIG_PATH))

if SAMPLER_CONFIG["file_type"] == "ZARR":
    ext = FILE_EXT_MAP[SAMPLER_CONFIG["file_type"]]
    CHIP_DATA_PATH = CHIP_DATA_PATH + f".{ext}"
    ANNO_DATA_PATH = ANNO_DATA_PATH + f".{ext}"
