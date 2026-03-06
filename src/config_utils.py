import importlib
import os
import yaml

from typing import Optional


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
