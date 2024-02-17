import torch
import torch.nn as nn
import numpy as np
import lightning as L


from transformers import VideoMAEConfig, VideoMAEModel, VideoMAEForPreTraining
from settings import SUNDIAL as configs


class Sundial(L.LightningModule):
    def __init__(self,
                 num_channels: int = configs["num_channels"],
                 num_frames: int = configs["num_frames"],
                 ):
        super().__init__()
        self.config = VideoMAEConfig(
            num_channels=num_channels,
            num_frames=num_frames,
        )
        self.back_bone = VideoMAEForPreTraining(self.config)

    def forward(self, inputs) -> torch.Tensor:
        return self.back_bone(inputs)

    def training_step(self, batch, *args) -> torch.Tensor:
        outputs = self.back_bone(batch)
        return outputs.loss

    def validation_step(self, batch, *args) -> torch.Tensor:
        outputs = self.back_bone(batch)
        return outputs.loss

    def predict_step(self, batch, *args) -> torch.Tensor:
        outputs = self.back_bone(batch)
        return outputs.loss

    def configure_optimizers(self):
        # TODO: implement dynamic learning rate
        return torch.optim.AdamW(self.model.parameters(), lr=1e-3)
