import lightning as L
import torch


class PrithviFCNCallbacks(L.Callback):
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

    def on_test_batch_end(self,
                          trainer: L.pytorch.trainer.trainer,
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

            # add each band separately
            for j in range(chip.shape[1]):
                band = chip[j, :, :, :]
                band = band.unsqueeze(1)
                pl_module.logger.experiment.add_images(
                    tag=f"samp-{index}_b-{j}_{batch_idx:03d}-{i:03d}-{dataloader_idx:03d}_chip",
                    img_tensor=band,
                    dataformats="NCHW"
                )

            pred = annotations[i]
            pred = pred.unsqueeze(1)
            pl_module.logger.experiment.add_images(
                tag=f"samp-{index}_{batch_idx:03d}-{i:03d}-{dataloader_idx:03d}_anno",
                img_tensor=pred,
                dataformats="NCHW"
            )

            logit = logits[i]
            logit = logit.unsqueeze(1)
            pl_module.logger.experiment.add_images(
                tag=f"samp-{index}_{batch_idx:03d}-{i:03d}-{dataloader_idx:03d}_pred",
                img_tensor=logit,
                dataformats="NCHW"
            )


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
