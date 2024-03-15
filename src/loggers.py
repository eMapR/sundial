from lightning.pytorch.loggers import TensorBoardLogger


class TBLogger(TensorBoardLogger):
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)
