import lightning as L


class SundialPLBase(L.LightningModule):
    def forward(self, chips):
        tokens = self.backbone(chips)
        features = self.neck(tokens)
        logits = self.head(features)

        return logits

    def training_step(self, batch):
        chips, annotations, _ = batch
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        return {"loss": loss}

    def validation_step(self, batch):
        chips, annotations, _ = batch
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        # reactivating logits for metric logging
        classes = self.activation(logits)

        return {"loss": loss, "classes": classes}

    def test_step(self, batch):
        chips, annotations, _ = batch
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        # reactivating logits for metric logging
        classes = self.activation(logits)

        return {"loss": loss, "classes": classes}

    def predict_step(self, batch):
        chips, _ = batch
        logits = self(chips)
        classes = self.activation(logits)

        return {"classes": classes}
