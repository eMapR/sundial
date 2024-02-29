import os
import torch

from lightning.pytorch.callbacks import BasePredictionWriter

from pipeline.settings import PY_WRITER as configs


class PredictionWriter(BasePredictionWriter):
    def __init__(self,
                 output_dir: str = configs["output_dir"],
                 write_interval: str = configs["write_interval"]):
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

        for i in range(len(prediction)):
            pred_image = prediction[i]
            orig_image = batch[0][i]
            image_name = batch[1][i]
            torch.save(pred_image,
                       os.path.join(out_path, f"{image_name}.pt"))
