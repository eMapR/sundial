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
        loss = self.criterion(logits, batch["anno"])
        
        output = self.activation(logits)
        
        return {"loss": loss, "output": output.detach()}

    def test_step(self, batch):
        logits = self(batch)
        loss = self.criterion(logits, batch["anno"])
        
        output = self.activation(logits)
        
        return {"loss": loss, "output": output.detach()}

    def predict_step(self, batch):
        logits = self(batch)
        output = self.activation(logits)

        return {"output": output.detach()}
