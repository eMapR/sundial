import lightning as L
import torch


class SundialPrithviCallback(L.Callback):
    def on_sanity_check_end(self, trainer, pl_module):
        img_size = pl_module.prithvi_params["model_args"]["img_size"]
        in_chans = pl_module.prithvi_params["model_args"]["in_chans"]
        num_frames = pl_module.prithvi_params["model_args"]["num_frames"]
        sample_img = torch.rand(
            (1, in_chans, num_frames, img_size, img_size)).cuda()
        pl_module.logger.experiment.add_graph(
            pl_module, sample_img, False, False)

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
        )

    def on_test_batch_end(self,
                          trainer: L.pytorch.trainer.trainer,
                          pl_module: L.LightningModule,
                          outputs: torch.Tensor,
                          batch: torch.Tensor,
                          batch_idx: int,
                          dataloader_idx: int = 0):
        _, annotations = batch
        loss = outputs["loss"]
        image = outputs["image"]
        logits = outputs["logits"]

        pl_module.log(
            name="test_loss",
            value=loss,
        )

        for i in range(image.shape[0]):
            chip = image[i]
            chip.squeeze(4)
            pl_module.logger.experiment.add_images(
                tag=f"chip_{batch_idx:02d}_{i:02d}",
                img_tensor=chip,
            )

            pred, _ = torch.max(annotations[i], dim=0, keepdim=True)
            pl_module.logger.experiment.add_image(
                tag=f"annotation_{batch_idx:02d}_{i:02d}",
                img_tensor=pred,
                dataformats="CHW"
            )

            logt = logits[i].unsqueeze(1)
            pl_module.logger.experiment.add_images(
                tag=f"prediction_{batch_idx:02d}_{i:02d}",
                img_tensor=logt,
                dataformats="NCHW"
            )
