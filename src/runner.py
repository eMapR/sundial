import comet_ml
import os
import shutil
import torch
import tarfile

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint

from callbacks import *
from dataloaders import *
from loss import *
from models import *
from utils import get_best_ckpt, tensors_to_tifs

from pipeline.utils import function_timer
from pipeline.sampler import (
    sample,
    annotate,
    download,
    calculate,
)
from pipeline.settings import (
    load_config,
    METHOD,
    RANDOM_STATE,
    CONFIG_PATH,
    META_DATA_PATH,
    CHECKPOINT_PATH,
    PREDICTION_PATH,
    SAMPLER_CONFIG,
    LOGGER_CONFIG,
    CHECKPOINT_CONFIG,
    EXPERIMENT_NAME
)


def train():
    # setting lower precision for GH200/cuda gpus. Not necessary because it is reset via configs but gets rid of the warning.
    torch.set_float32_matmul_precision("high")

    # setting up trainer defaults w/ paths from pipeline.settings
    base_config_path = os.path.join(CONFIG_PATH, "base.yaml")
    run_config_path = os.path.join(CONFIG_PATH, f"{METHOD}.yaml")
    args = [METHOD,
            f"--config={base_config_path}",
            f"--config={run_config_path}"]

    trainer_defaults = {
        "accelerator": "cuda",
        "callbacks": [
            LogSetupCallback(),
        ],
        "log_every_n_steps": 16,
        "logger": [
            {
                "class_path": "lightning.pytorch.loggers.CometLogger",
                "init_args": LOGGER_CONFIG
            }
        ],
        "enable_progress_bar": True,
        "profiler": "simple"
    }

    # setting up default callbacks and ckpts for methods
    match METHOD:
        case "fit":
            trainer_defaults["callbacks"].extend(
                [ModelCheckpoint(**CHECKPOINT_CONFIG)])
            trainer_defaults["logger"][0]["init_args"] |= {
                "auto_histogram_weight_logging": True,
                "auto_histogram_gradient_logging": True,
                "auto_histogram_activation_logging": True
            }
        case "test" | "predict":
            config = load_config(run_config_path)
            trainer_defaults["callbacks"].extend(
                [LoadCheckPointStrictFalse()])
            if "ckpt_path" not in config.keys() or config["ckpt_path"] is None:
                ckpt_path = get_best_ckpt(CHECKPOINT_PATH)
                args.append(f"--ckpt_path={ckpt_path}")

    LightningCLI(
        seed_everything_default=RANDOM_STATE,
        args=args,
        trainer_defaults=trainer_defaults,
        save_config_kwargs={"overwrite": True}
    )


@function_timer
def package():
    out_path = os.path.join("/tmp",
                            EXPERIMENT_NAME)
    os.makedirs(out_path)
    tensors_to_tifs(
        PREDICTION_PATH,
        out_path,
        META_DATA_PATH,
        SAMPLER_CONFIG["num_workers"]
    )
    tar_path = os.path.join(os.path.expanduser("~"),
                            EXPERIMENT_NAME)
    with tarfile.open(f"{tar_path}.tar.gz", "w:gz") as tar:
        tar.add(out_path, arcname=os.path.basename(out_path))
    shutil.rmtree(out_path)


if __name__ == "__main__":
    match METHOD:
        case "sample":
            sample()
        case "annotate":
            annotate()
        case "download":
            download()
        case "calculate":
            calculate()
        case "fit" | "validate" | "test" | "predict":
            train()
        case "package":
            package()
