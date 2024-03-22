import lightning as L
import os
import torch

from pipeline.settings import PREDICTION_PATH


class PrithviFCNCallbacks(L.Callback):
    def on_train_batch_end(self,
                           trainer: L.Trainer,
                           pl_module: L.LightningModule,
                           outputs: torch.Tensor,
                           batch: torch.Tensor,
                           batch_idx: int,):
        pl_module.log(
            name="train_loss",
            value=outputs["loss"],
            prog_bar=True,
        )

    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: torch.Tensor,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        pl_module.log(
            name="val_loss",
            value=outputs["loss"],
            prog_bar=True,
            sync_dist=True
        )

    def on_test_batch_end(self,
                          trainer: L.Trainer,
                          pl_module: L.LightningModule,
                          outputs: torch.Tensor,
                          batch: torch.Tensor,
                          batch_idx: int,
                          dataloader_idx: int = 0):
        chips, annotations, indices = batch
        loss = outputs["loss"]
        logits = outputs["logits"]

        pl_module.log(
            name="test_loss",
            value=loss,
            on_step=True
        )

        for i in range(chips.shape[0]):
            index = indices[i]
            chip = chips[i]

            # save each band separately
            # TODO: unnormalize the bands
            for j in range(chip.shape[1]):
                band = chip[j, :, :, :]
                band = band.unsqueeze(1)
                pl_module.logger.experiment.add_images(
                    tag=f"{index:07d}_b{j}_chip",
                    img_tensor=band,
                    dataformats="NCHW"
                )

            # save annotations and predictions
            pred = annotations[i]
            pred = pred.unsqueeze(1)
            pl_module.logger.experiment.add_images(
                tag=f"{index:07d}_anno",
                img_tensor=pred,
                dataformats="NCHW"
            )

            # save logits generated from model
            logit = logits[i]
            logit = logit.unsqueeze(1)
            pl_module.logger.experiment.add_images(
                tag=f"{index:07d}_pred",
                img_tensor=logit,
                dataformats="NCHW"
            )

    def on_predict_batch_end(self,
                             trainer: L.Trainer,
                             pl_module: L.LightningModule,
                             outputs: torch.Tensor,
                             batch: torch.Tensor,
                             batch_idx: int,
                             dataloader_idx: int = 0) -> None:
        _, indices = batch
        classes = outputs["classes"]
        for i in range(classes.shape[0]):
            index = indices[i]
            pred = classes[i]
            path = os.path.join(PREDICTION_PATH, f"{index:07d}_pred.pt")
            torch.save(pred, path)


class PrithviCallbacks(L.Callback):
    def on_train_batch_end(self,
                           trainer: L.pytorch.trainer.trainer,
                           pl_module: L.LightningModule,
                           outputs: torch.Tensor,
                           batch: torch.Tensor,
                           batch_idx: int,):
        pl_module.log(
            name="train_loss",
            value=outputs["loss"],
            prog_bar=True,
        )

    def on_validation_batch_end(self,
                                trainer: L.pytorch.trainer.trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: torch.Tensor,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        pl_module.log(
            name="val_loss",
            value=outputs["loss"],
            prog_bar=True,
            sync_dist=True
        )
