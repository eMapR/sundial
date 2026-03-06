import os

from config_utils import  load_yaml, recursive_merge
from constants import GEE_REQUEST_LIMIT, PIPELINE_CONFIG_PATH


# configs relating to sampler methods
PIPELINE_CONFIG = {
    ### Sampling settings
    
    "label_column": None,
    "date_column": None,
    "stat_actions": ["band_mean_stdv"],

    ### Image and downloadng settings

    "chunk_sizes": [1, 6, 224, 224],
    "scale": 30,
    "filter_intersect": False,
    "projection": "EPSG:5070",
    
    "ee_factory": {"class_path": "pipeline.ee_factories.LTMedoidImage",
                   "init_args": {}},
    
    ### MP and GEE specific settings
    
    "num_workers": GEE_REQUEST_LIMIT,
    "io_limit": 8,
}

# loading sampler config if it exists
if os.path.exists(PIPELINE_CONFIG_PATH):
    PIPELINE_CONFIG = recursive_merge(PIPELINE_CONFIG, load_yaml(PIPELINE_CONFIG_PATH))
