import os

from models import *
from dataloaders import *
from loggers import *
from writers import *

from lightning.pytorch.cli import ArgsType, LightningCLI

from pipeline.settings import RANDOM_STATE, CONFIG_PATH


def main(args: ArgsType = None):
    cli = LightningCLI(
        seed_everything_default=RANDOM_STATE,
        args=args,
    )


if __name__ == "__main__":
    method = os.getenv("SUNDIAL_METHOD")
    run_config_path = os.path.join(CONFIG_PATH, f"run.{method}.yaml")
    main([method, f"--config={run_config_path}"])
