import os
import torch

from lightning.pytorch.callbacks import BasePredictionWriter


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
        out_path = os.path.join(self.output_dir,
                                pl_module.__class__.__name__)
        os.makedirs(out_path, exist_ok=True)

        torch.save(prediction,
                   os.path.join(out_path, f"{batch_idx}.pred.pt"))
        torch.save(batch,
                   os.path.join(out_path, f"{batch_idx}.orig.pt"))
