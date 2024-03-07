import os
import torch

from lightning.pytorch.callbacks import BasePredictionWriter, ModelCheckpoint


class PredictionWriter(BasePredictionWriter):
    def __init__(self,
                 output_dir: str,
                 write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx
    ):
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(prediction,
                   os.path.join(self.output_dir, f"{batch_idx}.pred.pt"))
        torch.save(batch,
                   os.path.join(self.output_dir, f"{batch_idx}.orig.pt"))
