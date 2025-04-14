import torch
import lightning as L
from torch import nn


class SundialPLBase(L.LightningModule):
    def training_step(self, batch):
        loss = self.criterion(self(batch), batch["anno"])
        return {"loss": loss}

    def validation_step(self, batch):
        output = {"output": self(batch)}
        if self.criterion is not None:
            output["loss"] = self.criterion(output["output"], batch["anno"])
        if self.activation is not None:
            output["output"] = self.activation(output["output"])
        return output 

    def test_step(self, batch):
        output = {"output": self(batch)}
        if self.criterion is not None:
            output["loss"] = self.criterion(output["output"], batch["anno"])
        if self.activation is not None:
            output["output"] = self.activation(output["output"])
        return output 

    def predict_step(self, batch):
        output = {"output": self(batch)}
        if self.activation is not None:
            output["output"] = self.activation(output["output"])
        return output
