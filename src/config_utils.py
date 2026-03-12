import importlib
import os
import yaml

from typing import Optional

from constants import (
    CHECKPOINTS_PATH,
    CONFIG_PATH,
    PREDICTIONS_PATH,
    LOG_PATH,
    EXPERIMENT_SUFFIX,
    JOB_NAME,
    PIPELINE_CONFIG_PATH,
    SHAPE_NAME,
)


PACKAGE_CONFIG = {
    "format": os.getenv("SUNDIAL_PACKAGE_FORMAT")
}

RUN_CONFIG_DEFAULTS = {
    "model": None,
    "data": {
        "class_path": "dataloaders.generic_chips_dataset.GenericChipsDataModule",
        "init_args": {
            "batch_size": 32,
            "num_workers": 4,
            "sampler": {"class_path": "",
                        "init_args": {}},                                
            "dataloader_config": {},
            "static_transform_config": {"transforms": []},
            "dynamic_transform_config": {"transforms": []},
        }
    },
    "trainer": {
        "callbacks":[
            {
                "class_path": "callbacks.ModelSetupCallback"
            },
            {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": CHECKPOINTS_PATH,
                    "filename": "{epoch:04d_{val_loss:.3f}}",
                    "monitor": "val_loss",
                    "save_top_k": 3,
                    "save_last": False,
                    "auto_insert_metric_name": True,
                    "save_weights_only": False,
                    "every_n_epochs": 1,
                    "enable_version_counter": False
                    }
            }
        ]
    },
    "accelerator": "cuda",
    "log_every_n_steps": 16,
}


def save_yaml(config: dict, path: str | os.PathLike):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        return config if config else {}


def update_yaml(config: dict, path: str | os.PathLike) -> dict:
    if os.path.exists(path):
        old_config = load_yaml(path)
        config = recursive_merge(old_config, config)
    save_yaml(config, path)


def recursive_merge(dict1: dict, dict2: dict):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_merge(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def dynamic_import(loader: dict, kwargs: Optional[dict]=None):
    class_path = loader.get("class_path")
    init_args = loader.get("init_args", {})
    if kwargs is not None:
        init_args |= kwargs

    loader_path = class_path.rsplit(".", 1)
    module_path, class_name = loader_path
    loader_cls = getattr(importlib.import_module(module_path), class_name)
    
    return loader_cls(**init_args)


if __name__ == "__main__":
    from pipeline.settings import PIPELINE_CONFIG
    os.makedirs(CONFIG_PATH)
    os.makedirs(CHECKPOINTS_PATH)
    os.makedirs(PREDICTIONS_PATH)
    os.makedirs(LOG_PATH)

    config_path = os.path.join(CONFIG_PATH, "base.yaml")
    save_yaml(RUN_CONFIG_DEFAULTS, config_path)
    save_yaml(PIPELINE_CONFIG, PIPELINE_CONFIG_PATH)
