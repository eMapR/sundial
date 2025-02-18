import torch
import lightning as L
from torch import nn


class SundialPLBase(L.LightningModule):
    def training_step(self, batch):
        data = {k: v for k, v in batch.items() if k != "anno"}
        anno = batch["anno"]
        
        logits = self(data)
        loss = self.criterion(logits, anno)

        return {"loss": loss}

    def validation_step(self, batch):
        data = {k:v for k,v in batch.items() if k != "anno"}
        anno = batch["anno"]
        
        logits = self(data)
        loss = self.criterion(logits, anno)
        
        output = self.activation(logits)
        
        return {"loss": loss, "output": output.detach()}

    def test_step(self, batch):
        data = {k:v for k,v in batch.items() if k != "anno"}
        anno = batch["anno"]
        
        logits = self(data)
        loss = self.criterion(logits, anno)
        
        output = self.activation(logits)
        
        return {"loss": loss, "output": output.detach()}

    def predict_step(self, batch):
        data = {k:v for k,v in batch.items() if k != "anno"}
        
        logits = self(data)
        output = self.activation(logits)

        return {"output": output.detach()}
