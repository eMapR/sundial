import lightning as L
import os
import torch

from torchmetrics.functional.classification import binary_accuracy, binary_precision, binary_jaccard_index
from torchmetrics.functional.image import structural_similarity_index_measure
from typing import Any

from pipeline.settings import (load_config,
                               METHOD,
                               PREDICTION_PATH,
                               CHECKPOINT_PATH,
                               CONFIG_PATH,
                               LOG_PATH,
                               SAMPLER_CONFIG)
from utils import get_best_ckpt


class LoadCheckPointStrictFalse(L.Callback):
    def setup(self,
              trainer: L.Trainer,
              pl_module: L.LightningModule,
              stage: str):
        pl_module.strict_loading = False


class DrawPrithviONNXCallback(L.Callback):
    def setup(self,
              trainer: L.Trainer,
              pl_module: L.LightningModule,
              stage: str):
        d = pl_module.prithvi_params["model_args"]["num_frames"]
        c = pl_module.prithvi_params["model_args"]["in_chans"]
        h = pl_module.prithvi_params["model_args"]["img_size"]
        example_input = torch.randn((1, c, d, h, h),
                                    dtype=torch.float,
                                    device=pl_module.device)
        pl_module.to_onnx(LOG_PATH, example_input)


class LogSetupCallback(L.Callback):
    def setup(self,
              trainer: L.Trainer,
              pl_module: L.LightningModule,
              stage: str):
        base_config = load_config(os.path.join(CONFIG_PATH, "base.yaml"))
        meth_config = load_config(os.path.join(CONFIG_PATH, f"{METHOD}.yaml"))
        pl_module.logger.log_hyperparams(base_config)
        pl_module.logger.log_hyperparams(meth_config)
        pl_module.logger.log_hyperparams({"sampler": SAMPLER_CONFIG})

        # TODO: Add a global checkpoint file check
        if "ckpt_path" in meth_config.keys() and meth_config["ckpt_path"] is not None:
            ckpt_path = meth_config["ckpt_path"]
        elif METHOD == "test" or METHOD == "predict":
            ckpt_path = get_best_ckpt(CHECKPOINT_PATH)
        else:
            ckpt_path = trainer.ckpt_path
        pl_module.logger.log_hyperparams({"ckpt_path": ckpt_path})
        
        if trainer.lr_scheduler_configs:
            for c in trainer.lr_scheduler_configs:
                pl_module.logger.log_hyperparams(c)
        pl_module.save_hyperparameters()


class LogTrainCallback(L.Callback):
    def on_train_batch_end(self,
                           trainer: L.pytorch.trainer.trainer,
                           pl_module: L.LightningModule,
                           outputs: torch.Tensor,
                           batch: torch.Tensor,
                           batch_idx: int,):
        pl_module.log(
            name="train_loss",
            value=outputs["loss"],
            logger=True,
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
            logger=True,
            prog_bar=True,
            sync_dist=True
        )


class LogTrainExtCallback(L.Callback):
    def on_validation_batch_end(self,
                                trainer: L.pytorch.trainer.trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: torch.Tensor,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        _, annotations, _ = batch
        logits = outputs["logits"]

        # calculate metrics
        metrics = {
            "ssim": structural_similarity_index_measure(logits, annotations),
            "binary_jaccard_index": binary_jaccard_index(logits, annotations),
            "binary_precision": binary_precision(logits, annotations),
            "binary_accuracy": binary_accuracy(logits, annotations)
        }

        pl_module.log_dict(
            dictionary=metrics,
            logger=True,
            prog_bar=False,
            sync_dist=True
        )


class LogTestCallback(L.Callback):
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
            logger=True,
            prog_bar=False,
        )

        # unnormalize chips for loggings
        if trainer.test_dataloaders.dataset.means is not None and trainer.test_dataloaders.dataset.stds is not None:
            means = torch.tensor(
                trainer.test_dataloaders.dataset.means,
                dtype=torch.float,
                device=pl_module.device).view(-1, 1, 1, 1)
            stds = torch.tensor(
                trainer.test_dataloaders.dataset.stds,
                dtype=torch.float,
                device=pl_module.device).view(-1, 1, 1, 1)
            chips = chips * stds + means

        for i in range(chips.shape[0]):
            index = indices[i]
            chip = chips[i]
            anno = annotations[i]
            pred = logits[i]

            # save rgb and ir band separately
            for t in range(chip.shape[1]):
                image = chip[0:3, t, :, :].flip(0).permute(1, 2, 0)
                pl_module.logger.experiment.log_image(
                    image_data=image.detach().cpu(),
                    name=f"{index:07d}_rgb_t{t}_chip",
                )
                image = chip[3:6, t, :, :].flip(0).permute(1, 2, 0)
                pl_module.logger.experiment.log_image(
                    image_data=image.detach().cpu(),
                    name=f"{index:07d}_ir_t{t}_chip",
                )

            # save original annotations
            for c in range(anno.shape[0]):
                image = anno[c].unsqueeze(-1)
                pl_module.logger.experiment.log_image(
                    image_data=image.detach().cpu(),
                    name=f"{index:07d}_c{c+1}_anno",
                )

            # save logits generated from model
            for c in range(pred.shape[0]):
                image = pred[c].unsqueeze(-1)
                pl_module.logger.experiment.log_image(
                    image_data=image.detach().cpu(),
                    name=f"{index:07d}_c{c+1}_pred",
                )


class SaveTestCallback(L.Callback):
    def on_test_batch_end(self,
                          trainer: L.Trainer,
                          pl_module: L.LightningModule,
                          outputs: torch.Tensor,
                          batch: torch.Tensor,
                          batch_idx: int,
                          dataloader_idx: int = 0):
        chips, annotations, indices = batch
        logits = outputs["logits"]

        # unnormalize chips for loggings
        if trainer.test_dataloaders.dataset.means is not None and trainer.test_dataloaders.dataset.stds is not None:
            means = torch.tensor(
                trainer.test_dataloaders.dataset.means,
                dtype=torch.float,
                device=pl_module.device).view(-1, 1, 1, 1)
            stds = torch.tensor(
                trainer.test_dataloaders.dataset.stds,
                dtype=torch.float,
                device=pl_module.device).view(-1, 1, 1, 1)
            chips = chips * stds + means

        for i in range(chips.shape[0]):
            index = indices[i]
            chip = chips[i]
            anno = annotations[i]
            pred = logits[i]

            # save rgb and ir bands separately
            for t in range(chip.shape[1]):
                image = chip[0:3, t, :, :]
                path = os.path.join(
                    PREDICTION_PATH, f"{index:07d}_rgb_t{t}_chip.pt")
                torch.save(image, path)

                image = chip[3:6, t, :, :]
                path = os.path.join(
                    PREDICTION_PATH, f"{index:07d}_ir_t{t}_chip.pt")
                torch.save(image, path)

            # save annotations
            path = os.path.join(PREDICTION_PATH, f"{index:07d}_anno.pt")
            torch.save(anno, path)

            # save logits generated from model
            path = os.path.join(PREDICTION_PATH, f"{index:07d}_pred.pt")
            torch.save(pred, path)

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
