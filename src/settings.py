import os

from pipeline.settings import (
    save_yaml,
    CHECKPOINT_PATH,
    CONFIG_PATH,
    EXPERIMENT_SUFFIX,
    FILE_EXT_MAP,
    LOG_PATH,
    JOB_NAME,
    SAMPLE_CONFIG_PATH,
    SAMPLER_CONFIG,
    SAMPLE_NAME,
)

# default package settings
PACKAGE_CONFIG = {
    "format": os.getenv("SUNDIAL_PACKAGE_FORMAT")
}

# default lightning dataloader settings
DATALOADER_CONFIG = {
    "batch_size": 32,
    "num_workers": 16,
    "chip_size": SAMPLER_CONFIG["pixel_edge_size"],
    "time_step": SAMPLER_CONFIG["time_step"],
    "file_type": FILE_EXT_MAP[SAMPLER_CONFIG["file_type"]],
    "split_tif": None,
    "start_idx": None,
    "extension_config": {"extensions": []},
    "static_transform_config": {"transforms": []},
    "dynamic_transform_config": {"transforms": []},
}

# default lightning model checkpoint save settings
CHECKPOINT_CONFIG = {
    "dirpath": CHECKPOINT_PATH,
    "filename": "{epoch:04d}",
    "monitor": "val_loss",
    "save_top_k": 4,
    "auto_insert_metric_name": True,
    "save_weights_only": False,
    "every_n_epochs": 1,
    "enable_version_counter": True
}

# default lightning logger settings
LOGGER_CONFIG = {
    "api_key": os.getenv("COMET_API_KEY"),
    "workspace": os.getenv("COMET_WORKSPACE"),
    "save_dir": LOG_PATH,
    "project_name": SAMPLE_NAME.replace("_", "-"),
    "experiment_name": JOB_NAME,
    "log_code": False,
    "auto_param_logging": False,
    "auto_metric_logging": False,
    "auto_metric_step_rate": 1,
    "log_git_metadata": False,
    "log_git_patch": False,
    "display_summary_level": 0
}

if __name__ == "__main__":
    run_config = {
        "model": None,
        "data": {
            "class_path": "GenericChipsDataModule",
            "init_args": DATALOADER_CONFIG
        }
    }
    config_path = os.path.join(CONFIG_PATH, f"base.yaml")
    save_yaml(run_config, config_path)
    save_yaml(SAMPLER_CONFIG, SAMPLE_CONFIG_PATH)
