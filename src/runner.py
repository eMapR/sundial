import comet_ml
import os
import shutil
import torch
import tarfile

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint

from callbacks import *
from dataloaders import *
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
    load_yaml,
    METHOD,
    RANDOM_STATE,
    CONFIG_PATH,
    META_DATA_PATH,
    CHECKPOINT_PATH,
    PREDICTION_PATH,
    SAMPLER_CONFIG,
    LOGGER_CONFIG,
    CHECKPOINT_CONFIG,
    EXPERIMENT_NAME,
    EXPERIMENT_SUFFIX
)


class SundialCLI(LightningCLI):
    def add_arguments_to_parser(self,
                                parser: LightningArgumentParser):
        # placeholders to avoid parsing errors
        parser.add_argument("--criterion", default={}, type=dict)
        parser.add_argument("--activation", default={}, type=dict)
        parser.add_argument("--ckpt_monitor", default=None, type=str)


def train():
    # setting lower precision for GH200/cuda gpus to clear warning.
    torch.set_float32_matmul_precision("high")

    # setting up trainer defaults w/ paths from pipeline.settings
    base_config_path = os.path.join(CONFIG_PATH, "base.yaml")
    run_config_path = os.path.join(CONFIG_PATH, f"{METHOD}.yaml")
    run_configs = load_yaml(os.path.join(CONFIG_PATH, f"{METHOD}.yaml"))

    args = [METHOD,
            f"--config={base_config_path}",
            f"--config={run_config_path}"]

    # setting up default callbacks and ckpts for methods avoiding pL indirect
    trainer_defaults = {
        "accelerator": "cuda",
        "callbacks": [
            ModelSetupCallback(),
            DefineCriterionCallback(**run_configs.get("criterion", {})),
            DefineActivationCallback(**run_configs.get("activation", {})),
        ],
        "log_every_n_steps": 16,
        "logger": [
            {"class_path": "lightning.pytorch.loggers.CometLogger",
             "init_args": LOGGER_CONFIG}
        ],
        "enable_progress_bar": True,
        "profiler": "simple"
    }

    # setting method specific callbacks
    match METHOD:
        case "fit":
            if ckpt_monitor := run_configs.get("ckpt_monitor"):
                CHECKPOINT_CONFIG["monitor"] = ckpt_monitor
            trainer_defaults["callbacks"].extend([
                ModelCheckpoint(**CHECKPOINT_CONFIG)
            ]),
        case "test" | "predict":
            if run_configs.get("ckpt_path") is None:
                ckpt_path = get_best_ckpt(CHECKPOINT_PATH, EXPERIMENT_SUFFIX)
                args.append(f"--ckpt_path={ckpt_path}")

    SundialCLI(
        seed_everything_default=RANDOM_STATE,
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
