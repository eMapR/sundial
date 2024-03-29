import os
import torch

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint

from callbacks import *
from dataloaders import *
from loss import *
from models import *
from utils import get_best_ckpt

from pipeline.settings import (
    load_config,
    RANDOM_STATE,
    CONFIG_PATH,
    CHECKPOINT_PATH,
    LOGGER_CONFIG,
    CHECKPOINT_CONFIG,
)


def main(method: Literal["fit", "validate", "test", "predict"]):
    # setting lower precision for GH200/cuda gpus. Not necessary because it is reset via configs but gets rid of the warning.
    torch.set_float32_matmul_precision("high")

    # setting up trainer defaults w/ paths from pipeline.settings
    run_config_path = os.path.join(CONFIG_PATH, f"{method}.yaml")
    args = [method,
            f"--config={run_config_path}"]

    trainer_defaults = {
        "accelerator": "cuda",
        "log_every_n_steps": 16,
        "logger": [
            {
                "class_path": "lightning.pytorch.loggers.CometLogger",
                "init_args": LOGGER_CONFIG
            }
        ],
        "enable_progress_bar": True
    }

    # setting up default callbacks for fit method
    match method:
        case "fit":
            trainer_defaults["callbacks"] = [
                ModelCheckpoint(**CHECKPOINT_CONFIG),
            ]
        case "test" | "predict":
            config = load_config(run_config_path)
            if "ckpt_path" not in config.keys() or config["ckpt_path"] is None:
                ckpt_path = get_best_ckpt(CHECKPOINT_PATH)
            args.append(f"--ckpt_path={ckpt_path}")

    LightningCLI(
        seed_everything_default=RANDOM_STATE,
        args=args,
        trainer_defaults=trainer_defaults,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    method = os.getenv("SUNDIAL_METHOD")
    match os.getenv("SUNDIAL_METHOD"):
        case "sample":
            from pipeline.sampler import sample
            sample()
        case "annotate":
            from pipeline.sampler import annotate
            annotate()
        case "download":
            from pipeline.sampler import download
            download()
        case "calculate":
            from pipeline.sampler import calculate
            calculate()
        case "fit" | "validate" | "test" | "predict":
            main(method)
