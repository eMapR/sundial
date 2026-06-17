import torch
import lightning as L

from torch import nn


class SundialPLBase(L.LightningModule):
    def __init__(self, criterion=None, activation=None):
        super().__init__()
        self.criterion = criterion
        self.activation = activation
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for k in batch.keys():
            if k != "meta":
                batch[k] = batch[k].to(device)
        return batch
    
    def training_step(self, batch):
        loss = self.criterion(self(batch), batch["target"])
        return {"loss": loss}

    def validation_step(self, batch):
        output = {"output": self(batch)}
        if self.criterion is not None:
            output["loss"] = self.criterion(output["output"], batch["target"])
        if self.activation is not None:
            output["output"] = self.activation(output["output"])
        return output 

    def test_step(self, batch):
        output = {"output": self(batch)}
        if self.criterion is not None:
            output["loss"] = self.criterion(output["output"], batch["target"])
        if self.activation is not None:
            output["output"] = self.activation(output["output"])
        return output 

    def predict_step(self, batch):
        output = {"output": self(batch)}
        if self.activation is not None:
            output["output"] = self.activation(output["output"])
        return output
