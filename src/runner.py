import os
import torch

from lightning.pytorch.cli import ArgsType, LightningCLI

from models import *
from dataloaders import *
from loggers import *
from writers import *
from pipeline.settings import RANDOM_STATE, CONFIG_PATH


def main(args: ArgsType = None):
    torch.set_float32_matmul_precision("high")
    LightningCLI(
        seed_everything_default=RANDOM_STATE,
        args=args,
    )


if __name__ == "__main__":
    method = os.getenv("SUNDIAL_METHOD")
    run_config_path = os.path.join(CONFIG_PATH, f"{method}.yaml")
    match method:
        case "sample":
            from pipeline.sampler import main
            main()
        case "download":
            from pipeline.downloader import main
            main()
        case "fit" | "validate" | "test" | "predict":
            main([method, f"--config={run_config_path}"])
