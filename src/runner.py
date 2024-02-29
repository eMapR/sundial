from lightning.pytorch.cli import LightningCLI

from models import *
from dataloaders import *
from pipeline.settings import RANDOM_STATE
from writers import *


def main():
    LightningCLI(
        seed_everything_default=RANDOM_STATE,
    )


if __name__ == "__main__":
    main()
