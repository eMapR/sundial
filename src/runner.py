import comet_ml
import os
import torch
import tarfile

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from callbacks import DefineActivationCallback, DefineCriterionCallback, LogSetupCallback, ModelSetupCallback
from settings import CHECKPOINT_CONFIG, EARLY_STOPPING_CONFIG, LOGGER_CONFIG, PACKAGE_CONFIG
from utils import get_best_ckpt, get_latest_ckpt, tensors_to_tifs

from pipeline.utils import function_timer
from pipeline.sampler import (
    sample,
    annotate,
    download,
    calculate,
    index,
)
from pipeline.settings import (
    load_yaml,
    BASE_CONFIG_PATH,
    CHECKPOINT_PATH,
    EXPERIMENT_FULL_NAME,
    EXPERIMENT_SUFFIX,
    META_DATA_PATH,
    METHOD,
    METHOD_CONFIG_PATH,
    PREDICTION_PATH,
    RANDOM_SEED,
    SAMPLER_CONFIG,
)


class SundialCLI(LightningCLI):
    def add_arguments_to_parser(self,
                                parser: LightningArgumentParser):
        # placeholders to avoid parsing errors
        parser.add_argument("--criterion", default={}, type=dict)
        parser.add_argument("--activation", default={}, type=dict)
        parser.add_argument("--early_stopping", default={}, type=dict)
        parser.add_argument("--model_checkpoint", default={}, type=dict)


def train():
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
        "callbacks": [
            ModelSetupCallback(),
            DefineCriterionCallback(**run_configs.get("criterion", {})),
            DefineActivationCallback(**run_configs.get("activation", {})),
        ],
        "log_every_n_steps": 16,
        "enable_progress_bar": False,
    }

    # set up comet logger if no logger is specified
    if not run_configs["trainer"].get("logger"):
        trainer_defaults["logger"] = {
            "class_path": "lightning.pytorch.loggers.CometLogger",
            "init_args": LOGGER_CONFIG}

    # setting method specific callbacks
    match METHOD:
        case "fit":
            model_checkpoint = CHECKPOINT_CONFIG
            if run_model_checkpoint := run_configs.get("model_checkpoint"):
                model_checkpoint |= run_model_checkpoint
            early_stopping = EARLY_STOPPING_CONFIG
            if run_early_stopping := run_configs.get("early_stopping"):
                early_stopping |= run_early_stopping
            trainer_defaults["callbacks"].extend([
                ModelCheckpoint(**model_checkpoint),
                EarlyStopping(**early_stopping),
            ]),
        case "test" | "predict":
            ckpt_path = run_configs.get("ckpt_path", False)
            match ckpt_path:
                case "best":
                    ckpt_path = get_best_ckpt(
                        CHECKPOINT_PATH, EXPERIMENT_SUFFIX)
                    args.append(f"--ckpt_path={ckpt_path}")
                case "latest":
                    ckpt_path = get_latest_ckpt(
                        CHECKPOINT_PATH, EXPERIMENT_SUFFIX)
                case False:
                    ckpt_path = get_best_ckpt(
                        CHECKPOINT_PATH, EXPERIMENT_SUFFIX)
                case None:
                    pass
                case _:
                    args.append(
                        f"--ckpt_path={os.path.join(CHECKPOINT_PATH, ckpt_path)}")

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
    tensors_dir_path = tensors_to_tifs(
        PREDICTION_PATH,
        EXPERIMENT_FULL_NAME,
        META_DATA_PATH,
        SAMPLER_CONFIG["num_workers"]
    )
    match PACKAGE_CONFIG["format"]:
        case "tar":
            tar_path = os.path.join(
                os.path.expanduser("~"), EXPERIMENT_FULL_NAME)
            with tarfile.open(f"{tar_path}.tar.gz", "w:gz") as tar:
                tar.add(tensors_dir_path, arcname=EXPERIMENT_FULL_NAME)


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
        case "index":
            index()
        case "fit" | "validate" | "test" | "predict":
            train()
        case "package":
            package()
