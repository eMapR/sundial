import lightning as L
import torch


class SundialPrithviCallback(L.Callback):
    def on_train_batch_end(self,
                           trainer,
                           pl_module,
                           outputs,
                           batch,
                           batch_idx):
        pl_module.log(
            name="train_loss",
            value=outputs,
            prog_bar=True,
        )

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx):
        pl_module.log(
            name="val_loss",
            value=outputs,
        )

    def on_test_batch_end(self,
                          trainer,
                          pl_module,
                          outputs,
                          batch,
                          batch_idx,
                          dataloader_idx):
        _, annotations = batch
        loss, image, logits = outputs

        pl_module.log(
            name="test_loss",
            value=loss,
        )

        for i in range(image.shape[0]):
            vid = image[i].unsqueeze(0)
            pl_module.logger.experiment.add_video(
                tag="chips",
                vid_tensor=vid,
                fps=1,
            )

            pred, _ = torch.max(annotations[i], dim=0, keepdim=True)
            pl_module.logger.experiment.add_image(
                tag="annotations",
                img_tensor=pred,
                dataformats="CHW"
            )

            logt = logits[i].unsqueeze(1)
            pl_module.logger.experiment.add_images(
                tag="predictions",
                img_tensor=logt,
                dataformats="NCHW"
            )
