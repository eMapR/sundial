import os


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