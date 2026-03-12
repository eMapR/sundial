import os

# experiment information
SHAPE_NAME = os.getenv("SUNDIAL_SHAPE_NAME")
EXPERIMENT_PREFIX = os.getenv("SUNDIAL_EXPERIMENT_PREFIX")
EXPERIMENT_SUFFIX = os.getenv("SUNDIAL_EXPERIMENT_SUFFIX")
EXPERIMENT_BASE_NAME = os.getenv("SUNDIAL_EXPERIMENT_BASE_NAME")
EXPERIMENT_FULL_NAME = os.getenv("SUNDIAL_EXPERIMENT_FULL_NAME")
METHOD = os.getenv("SUNDIAL_METHOD")
JOB_NAME = f"{METHOD}_{EXPERIMENT_FULL_NAME}"

# base paths
BASE_PATH = os.getenv("SUNDIAL_BASE_PATH")
DATA_PATH = os.path.join(BASE_PATH, "data")
SHAPES_PATH = os.path.join(BASE_PATH, "shapes")
EXPERIMENTS_PATH = os.path.join(BASE_PATH, "experiments")

# experiment paths
EXPERIMENT_PATH = os.path.join(EXPERIMENTS_PATH, EXPERIMENT_BASE_NAME)
CHECKPOINTS_PATH = os.path.join(EXPERIMENT_PATH, EXPERIMENT_SUFFIX, "checkpoints")
PREDICTIONS_PATH = os.path.join(EXPERIMENT_PATH, EXPERIMENT_SUFFIX, "predictions")
OUTPUT_DATA_PATH = os.path.join(EXPERIMENT_PATH, EXPERIMENT_SUFFIX, "output_data")
CONFIG_PATH = os.path.join(EXPERIMENT_PATH, "configs")
LOG_PATH = os.path.join(EXPERIMENT_PATH, "logs")

# config paths
PIPELINE_CONFIG_PATH = os.path.join(CONFIG_PATH, "pipeline.yaml")
METHOD_CONFIG_PATH = os.path.join(CONFIG_PATH, f"{METHOD}.yaml")
EXPERIMENT_CONFIG_PATH = os.path.join(CONFIG_PATH, METHOD, f"{EXPERIMENT_SUFFIX}.yaml")
BASE_CONFIG_PATH = os.path.join(CONFIG_PATH, "base.yaml")

# sample and data paths
STAT_DATA_PATH = os.path.join(EXPERIMENT_PATH, "stat_data.yaml")
IMAGERY_PATH = os.path.join(EXPERIMENT_PATH, "imagery")
ANNOTATIONS_PATH = os.path.join(EXPERIMENT_PATH, "annotations")

# shapefile and source data paths
GEO_PROC_PATH = os.path.join(SHAPES_PATH, SHAPE_NAME)

# non configurable GEE, image, and meta data settings
RANDOM_SEED = 42
IDX_NAME_ZFILL = 8
GEE_REQUEST_LIMIT = 42
GEE_REQUEST_LIMIT_MB = 50331648
EE_END_POINT = 'https://earthengine-highvolume.googleapis.com'
