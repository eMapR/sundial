import time

from lightning.pytorch.cli import ArgsType, LightningCLI

from datamodule import ChipsDataModule

from settings import RANDOM_STATE
from sundial import Sundial


def main(args: ArgsType = None, **kwargs):
    # TODO: implement downloader and sampler pipelines as subcommands
    LightningCLI(
        model_class=Sundial,
        seed_everything_default=RANDOM_STATE,
        datamodule_class=ChipsDataModule,
        args=args,
    )


if __name__ == "__main__":
    main()
