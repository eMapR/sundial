from lightning.pytorch.loggers import TensorBoardLogger


class ExperimentLogger(TensorBoardLogger):
    def __init__(self,
                 save_dir: str,
                 name: str,
                 **kwargs):
        super().__init__(save_dir, name, **kwargs)
