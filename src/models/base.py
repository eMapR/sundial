import lightning as L


class SundialPLBase(L.LightningModule):
    def forward(self, chips):
        tokens = self.backbone(*chips)
        features = self.neck(tokens)
        logits = self.head(features)

        return logits

    def training_step(self, batch):
        chips = batch[0:1]
        annotations = batch[1]
        
        if len(batch) > 3:
            chips += batch[3:]
        
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        return {"loss": loss}

    def validation_step(self, batch):
        chips = batch[0:1]
        annotations = batch[1]
        
        if len(batch) > 3:
            chips += batch[3:]
        
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        # reactivating logits for metric logging
        output = self.activation(logits)

        return {"loss": loss, "output": output}

    def test_step(self, batch):
        chips = batch[0:1]
        annotations = batch[1]
        
        if len(batch) > 3:
            chips += batch[3:]
        
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        # reactivating logits for metric logging
        output = self.activation(logits)

        return {"loss": loss, "output": output}

    def predict_step(self, batch):
        chips = batch[0:1]
        
        if len(batch) > 3:
            chips += batch[3:]
        
        logits = self(chips)
        output = self.activation(logits)

        return {"output": output}
