import time

from lightning.pytorch.cli import LightningCLI
from sundial import Sundial
from datamodule import ChipsDataModule


def main(*args, **kwargs):
    cli = LightningCLI(
        model_class=Sundial,
        seed_everything_default=time.time(),
        datamodule_class=ChipsDataModule,
        args=args,
    )
    return cli


if __name__ == "__main__":
    main()
