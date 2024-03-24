import lightning as L
import os
import torch

from pipeline.logger import get_logger
from pipeline.settings import load_config, PREDICTION_PATH, META_DATA_PATH, LOG_PATH, SAMPLER_CONFIG, CONFIG_PATH
from utils import tensors_to_tifs

LOGGER = get_logger(LOG_PATH, os.getenv("SUNDIAL_METHOD"))


class PrithviCallbacks(L.Callback):
     def setup(self,
               trainer: L.Trainer,
               pl_module: L.LightningModule,
               stage: str):
         method = os.getenv("SUNDIAL_METHOD")
         run_config_path = os.path.join(CONFIG_PATH, f"{method}.yaml")
         config = load_config(run_config_path)
         pl_module.logger.log_hyperparams(config)

     def on_train_batch_end(self,
                            trainer: L.pytorch.trainer.trainer,
                            pl_module: L.LightningModule,
                            outputs: torch.Tensor,
                            batch: torch.Tensor,
                            batch_idx: int,):
         pl_module.log(
             name="train_loss",
             value=outputs["loss"],
             logger=False,
             prog_bar=True,
         )
         pl_module.logger.log_metrics(
             metrics={"train_loss": outputs["loss"]},
             step=trainer.global_step
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
             logger=False,
             prog_bar=True,
             sync_dist=True
         )
         pl_module.logger.log_metrics(
             metrics={"val_loss": outputs["loss"]},
             step=trainer.global_step
         )

class PrithviFCNCallbacks(PrithviCallbacks):
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
            logger=False,
            prog_bar=False,
        )

        for i in range(chips.shape[0]):
            index = indices[i]
            chip = chips[i]

            # save each band separately
            # TODO: unnormalize the bands
            for b in range(chip.shape[1]):
                band = chip[b, :, :, :]
                for t in range(band.shape[0]):
                    pl_module.logger.experiment.log_image(
                        tag=f"{index:07d}_b{b}_t{t}_chip",
                        img_tensor=band[t].unsqueeze(0),
                        image_channels="first"
                    )

            # save annotations and predictions
            # TODO: flatten and overlay annotations onto chips
            pred = annotations[i]
            for c in range(pred.shape[0]):
                pl_module.logger.experiment.log_image(
                    tag=f"{index:07d}_c{c}_anno",
                    img_tensor=pred[c].unsqueeze(0),
                    image_channels="first"
                )

            # save logits generated from model
            # TODO: flatten and overlay predictions onto chips
            logit = logits[i]
            for c in range(logit.shape[0]):
                pl_module.logger.experiment.log_image(
                    tag=f"{index:07d}_c{c}_pred",
                    img_tensor=logit[c].unsqueeze(0),
                    image_channels="first"
                )

    def on_predict_batch_end(self,
                             trainer: L.Trainer,
                             pl_module: L.LightningModule,
                             outputs: torch.Tensor,
                             batch: torch.Tensor,
                             batch_idx: int,
                             dataloader_idx: int = 0):
        _, indices = batch
        classes = outputs["classes"]
        for i in range(classes.shape[0]):
            index = indices[i]
            pred = classes[i]
            path = os.path.join(PREDICTION_PATH, f"{index:07d}_pred.pt")
            torch.save(pred, path)

    def on_predict_end(self,
                       trainer: L.Trainer,
                       pl_module: L.LightningModule):
        LOGGER.info(
            f"Model prediction completed. Converting to tifs with spatial metadata. num_workers={SAMPLER_CONFIG['num_workers']}...")
        tensors_to_tifs(PREDICTION_PATH,
                        PREDICTION_PATH,
                        META_DATA_PATH,
                        SAMPLER_CONFIG['num_workers'])

