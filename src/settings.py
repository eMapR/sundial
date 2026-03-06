import os

from config_utils import save_yaml
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
from pipeline.settings import PIPELINE_CONFIG


# default package settings
PACKAGE_CONFIG = {
    "format": os.getenv("SUNDIAL_PACKAGE_FORMAT")
}

# default lightning dataloader settings
DATAMODULE_CONFIG = {
    "batch_size": 32,
    "num_workers": 4,                                       # number of workers to use for loading onto GPU
    "sampler": {"class_path": "",
                "init_args": {}},                                
    "dataloader_config": {},
    "static_transform_config": {"transforms": []},          # transforms defined as nn.Modules to perform sequentially. follows Pytorch Lightning format w/ class_path & init_args
                                                            # set "image_only" to true if the transforms should only be performed on image and not annotation
                                                            # transforms will be composed into sequential transformation
                                                            # set "methods" to list [METHOD NAMES] to specify which methods to perform transformations on
    "dynamic_transform_config": {"transforms": []},         # transforms defined as nn.Modules to augment dataset during training. follows Pytorch Lightning format w/ class_path & init_args
                                                            # set "targets" to list ["chip" | "anno" | "chip", "anno"] if the transforms should only be performed on image, annotation or both
                                                            # len(dataset) will be multiplied by number of transforms and each will be performed as its own sample
                                                            # set "include_original" to true to include an original sample in training without transformation
}

# default lightning model checkpoint save settings (See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
# can be set in any of the config files under "model_checkpoint key"
CHECKPOINT_CONFIG = {
    "dirpath": CHECKPOINTS_PATH,                             # directory path to save checkpoint files. will default to specification in settings.py
    "filename": "{epoch:04d}",
    "monitor": "val_loss",
    "save_top_k": 3,
    "save_last": False,
    "auto_insert_metric_name": True,
    "save_weights_only": False,
    "every_n_epochs": 1,
    "enable_version_counter": False
}

# default lightning logger settings (See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.comet.html#module-lightning.pytorch.loggers.comet)
# can be set in any of the config yamls under "trainer.logger" key
LOGGER_CONFIG = {
    "api_key": os.getenv("COMET_API_KEY"),
    "workspace": os.getenv("COMET_WORKSPACE"),
    "save_dir": LOG_PATH,
    "project_name": SHAPE_NAME.replace("_", "-"),
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
    os.makedirs(CONFIG_PATH)
    os.makedirs(CHECKPOINTS_PATH)
    os.makedirs(PREDICTIONS_PATH)
    os.makedirs(LOG_PATH)

    run_config = {
        "model": None,
        "data": {
            "class_path": "dataloaders.generic_chips_dataset.GenericChipsDataModule",
            "init_args": DATAMODULE_CONFIG
        }
    }
    config_path = os.path.join(CONFIG_PATH, "base.yaml")
    save_yaml(run_config, config_path)
    save_yaml(PIPELINE_CONFIG, PIPELINE_CONFIG_PATH)
