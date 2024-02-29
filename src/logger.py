from lightning.fabric.loggers import TensorBoardLogger

from pipeline.settings import TENSORBOARD_LOGGER as configs


class SundialLogger(TensorBoardLogger):
    def __init__(self,
                 root_dir: str = configs["root_dir"],
                 name: str = configs["name"],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_dir = root_dir
        self.name = name
