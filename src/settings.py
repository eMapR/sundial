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
    "preprocess_config": {"preprocesses": []},
    "transform_config": {"transforms": []},
}

# default early stopping settings
EARLY_STOPPING_CONFIG = {
    "monitor": "val_loss",
    "min_delta": 0.0,
    "patience": 128,
    "verbose": True,
    "mode": "min",
    "strict": True,
    "check_finite": True,
    "stopping_threshold": None,
    "divergence_threshold": None,
    "check_on_train_epoch_end": False,
    "log_rank_zero_only": False
}

# default lightning model checkpoint save settings
CHECKPOINT_CONFIG = {
    "dirpath": CHECKPOINT_PATH,
    "filename": "epoch-{epoch:04d}_val_loss-{val_loss:.3f}",
    "monitor": "val_loss",
    "save_top_k": 16,
    "auto_insert_metric_name": False,
    "save_weights_only": False,
    "every_n_epochs": 1,
    "enable_version_counter": True
}
if EXPERIMENT_SUFFIX:
    CHECKPOINT_CONFIG["filename"] += f"_{EXPERIMENT_SUFFIX}"

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
    "auto_metric_step_rate": 16,
    "log_git_metadata": False,
    "log_git_patch": False,
    "display_summary_level": 0
}

if __name__ == "__main__":
    run_config = {
                    "model": None,
                    "data": {
                        "class_path": "ChipsDataModule",
                        "init_args": DATALOADER_CONFIG
                    }
                }
    config_path = os.path.join(CONFIG_PATH, f"base.yaml")
    save_yaml(run_config, config_path)
    save_yaml(SAMPLER_CONFIG, SAMPLE_CONFIG_PATH)
