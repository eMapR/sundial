import comet_ml
import os
import shutil
import torch
import tarfile

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint

from callbacks import DefineActivationCallback, DefineCriterionCallback, LogSetupCallback, ModelSetupCallback
from settings import CHECKPOINT_CONFIG, LOGGER_CONFIG, PACKAGE_CONFIG
from utils import get_best_ckpt, get_latest_ckpt, tensors_to_tifs

from config_utils import load_yaml
from constants import (
    DATA_PATH,
    BASE_CONFIG_PATH,
    CHECKPOINT_PATH,
    EXPERIMENT_FULL_NAME,
    EXPERIMENT_SUFFIX,
    LOG_PATH,
    META_DATA_PATH,
    METHOD,
    METHOD_CONFIG_PATH,
    PREDICTION_PATH,
    RANDOM_SEED,
)
from pipeline.pipeline import (
    sample,
    annotate,
    download,
    index,
)
from pipeline.settings import SAMPLER_CONFIG
from pipeline.logging import function_timer, get_logger


LOGGER = get_logger(LOG_PATH, METHOD)


class SundialCLI(LightningCLI):
    def add_arguments_to_parser(self,
                                parser: LightningArgumentParser):
        # placeholders to avoid parsing errors
        parser.add_argument("--comet_ml", default=True, type=bool)
        parser.add_argument("--criterion", default={}, type=dict)
        parser.add_argument("--activation", default={}, type=dict)
        parser.add_argument("--model_checkpoint", default={}, type=dict)


def run():
    # setting lower precision for GH200/cuda gpus
    torch.set_float32_matmul_precision("high")

    # setting up trainer defaults w/ paths from pipeline.settings
    run_configs = load_yaml(METHOD_CONFIG_PATH)
    
    args = [METHOD,
            f"--config={BASE_CONFIG_PATH}",
            f"--config={METHOD_CONFIG_PATH}"]

    # setting up default callbacks and ckpts for methods avoiding pL indirect
    trainer_defaults = {
        "accelerator": "cuda",
        "log_every_n_steps": 16,
        "callbacks": [
            ModelSetupCallback(),
            DefineCriterionCallback(**run_configs.get("criterion", {})),
            DefineActivationCallback(**run_configs.get("activation", {})),
        ],
    }

    # set up comet logger if no logger is specified
    if run_configs.get("comet_ml", True):
        trainer_defaults["logger"] = {
            "class_path": "lightning.pytorch.loggers.CometLogger",
            "init_args": LOGGER_CONFIG}

    # setting method specific callbacks
    match METHOD:
        case "fit":
            model_checkpoint = CHECKPOINT_CONFIG
            if run_model_checkpoint := run_configs.get("model_checkpoint"):
                model_checkpoint |= run_model_checkpoint
            model_checkpoint["filename"] += f"_{{{model_checkpoint['monitor']}:.3f}}"
            if EXPERIMENT_SUFFIX:
                model_checkpoint["filename"] += f"_{EXPERIMENT_SUFFIX}"
            trainer_defaults["callbacks"].extend([
                ModelCheckpoint(**model_checkpoint),
            ]),
        case "test" | "predict":
            ckpt_path = run_configs.get("ckpt_path", False)
            match ckpt_path:
                case "best":
                    ckpt_path = get_best_ckpt(CHECKPOINT_PATH, EXPERIMENT_SUFFIX)
                case "latest":
                    ckpt_path = get_latest_ckpt(CHECKPOINT_PATH, EXPERIMENT_SUFFIX)
                case False:
                    ckpt_path = get_best_ckpt(CHECKPOINT_PATH, EXPERIMENT_SUFFIX)
                case None:
                    ckpt_path = "null"
                case _:
                    if CHECKPOINT_PATH not in ckpt_path:
                        ckpt_path = os.path.join(CHECKPOINT_PATH, ckpt_path)
            args.append(f"--ckpt_path={ckpt_path}")

    # using cli api instead of Trainer to avoid some code
    SundialCLI(
        seed_everything_default=RANDOM_SEED,
        args=args,
        trainer_defaults=trainer_defaults,
        save_config_callback=LogSetupCallback,
        save_config_kwargs={
            "config_filename": "config.yaml",
            "overwrite": True
        }
    )


@function_timer
def package():
    LOGGER.info(DATA_PATH)
    tensors_dir_path = tensors_to_tifs(
        os.path.join(PREDICTION_PATH, EXPERIMENT_SUFFIX),
        EXPERIMENT_FULL_NAME,
        DATA_PATH,
        META_DATA_PATH,
        SAMPLER_CONFIG["num_workers"],
        LOGGER,
    )


if __name__ == "__main__":
    match METHOD:
        case "sample":
            sample()
        case "annotate":
            annotate()
        case "download":
            download()
        case "index":
            index()
        case "fit" | "validate" | "test" | "predict":
            run()
        case "package":
            package()
