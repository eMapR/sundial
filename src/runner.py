import os
import shutil
import torch
import tarfile

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from utils import get_best_ckpt, get_latest_ckpt
from config_utils import load_yaml
from constants import (
    BASE_CONFIG_PATH,
    EXPERIMENT_FULL_NAME,
    EXPERIMENT_SUFFIX,
    EXPERIMENT_CONFIG_PATH,
    LOG_PATH,
    METHOD,
    METHOD_CONFIG_PATH,
    RANDOM_SEED,
)
from pipeline.pipeline import (
    annotate,
    download,
    stats,
)
from pipeline.settings import PIPELINE_CONFIG
from pipeline.logging import function_timer


class SundialCLI(LightningCLI):
    pass


def run():
    torch.set_float32_matmul_precision("high")
    args = [METHOD,
                    f"--config={BASE_CONFIG_PATH}",
                    f"--config={METHOD_CONFIG_PATH}"]
    if os.path.exists(EXPERIMENT_CONFIG_PATH):
        args.append(f"--config={EXPERIMENT_CONFIG_PATH}")    

    match METHOD:
        case "test" | "predict":
            run_configs = load_yaml(METHOD_CONFIG_PATH)
            ckpt_path = run_configs.get("ckpt_path", False)
            match ckpt_path:
                case "best":
                    ckpt_path = get_best_ckpt(CHECKPOINT_PATH)
                case "latest":
                    ckpt_path = get_latest_ckpt(CHECKPOINT_PATH)
                case False:
                    ckpt_path = get_best_ckpt(CHECKPOINT_PATH)
                case None:
                    ckpt_path = "null"
                case _:
                    if CHECKPOINT_PATH not in ckpt_path:
                        ckpt_path = os.path.join(CHECKPOINT_PATH, ckpt_path)
            args.append(f"--ckpt_path={ckpt_path}")

    SundialCLI(
        seed_everything_default=RANDOM_SEED,
        args=args,
        save_config_kwargs={
            "config_filename": "config.yaml",
            "overwrite": True
        },
        trainer_defaults={
            "logger": {
                "class_path": "lightning.pytorch.loggers.CSVLogger",
                "init_args": {
                    "name": EXPERIMENT_SUFFIX,
                    "save_dir": LOG_PATH,
                }    
            }
        }
    )


if __name__ == "__main__":
    match METHOD:
        case "download":
            download()
        case "annotate":
            annotate()
        case "stats":
            stats()
        case "fit" | "validate" | "test" | "predict":
            run()
