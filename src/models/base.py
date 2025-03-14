import torch
import lightning as L
from torch import nn


class SundialPLBase(L.LightningModule):
    def training_step(self, batch):
        logits = self(batch)
        loss = self.criterion(logits, batch["anno"])

        return {"loss": loss}

    def validation_step(self, batch):
        logits = self(batch)
        output = {"output": self.activation(logits).detach()} 
        if self.criterion is not None:
            output["loss"] = self.criterion(logits, batch["anno"])
        return output 

    def test_step(self, batch):
        logits = self(batch)
        output = {"output": self.activation(logits).detach()} 
        if self.criterion is not None:
            output["loss"] = self.criterion(logits, batch["anno"])
        return output

    def predict_step(self, batch):
        logits = self(batch)
        output = {"output": self.activation(logits).detach()} 
        return output
