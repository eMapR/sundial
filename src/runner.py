import time

from lightning.pytorch.cli import ArgsType, LightningCLI

from sundial import Sundial
from datamodule import ChipsDataModule


def main(args: ArgsType = None, **kwargs):
    # TODO: implement downloader and sampler pipelines as subcommands
    LightningCLI(
        model_class=Sundial,
        seed_everything_default=time.time(),
        datamodule_class=ChipsDataModule,
        args=args,
    )


if __name__ == "__main__":
    main()
