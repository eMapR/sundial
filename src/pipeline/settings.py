import os

from config_utils import  load_yaml, recursive_merge
from constants import GEE_REQUEST_LIMIT, PIPELINE_CONFIG_PATH


# configs relating to sampler methods
PIPELINE_CONFIG = {
    ### Sampling settings
    "annotator":{"class_path": "pipeline.annotators.XarrDateAnnotator",
                 "init_args": {
                    "label_column": None,
                    "date_column": None,       
                    }
    },
    "stats_actions": ["band_mean_stdv", "class_counts"],

    ### Image and downloadng settings
    "chunk_sizes": [6, 1, 224, 224],
    "dtype": "float32",
    "scale": 30,
    "filter_intersect": False,
    
    "ee_factory": {"class_path": "pipeline.ee_factories.LTMedoidImage",
                   "init_args": {}},
    
    ### MP and GEE specific settings
    
    "num_workers": GEE_REQUEST_LIMIT,
    "io_limit": 8,
}

# loading sampler config if it exists
if os.path.exists(PIPELINE_CONFIG_PATH):
    PIPELINE_CONFIG = recursive_merge(PIPELINE_CONFIG, load_yaml(PIPELINE_CONFIG_PATH))
