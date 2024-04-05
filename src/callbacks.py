import copy
import lightning as L
import os
import torch

from torchmetrics.functional.classification import (binary_accuracy,
                                                    binary_precision,
                                                    binary_jaccard_index,
                                                    multiclass_accuracy,
                                                    multiclass_precision,
                                                    multiclass_jaccard_index)
from torchmetrics import Dice, JaccardIndex
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.ops import sigmoid_focal_loss
from typing import Optional

from losses import (JacardLoss,
                    DiceLoss,
                    DiceBCELoss,
                    FocalLoss,
                    TverskyLoss,
                    FocalTverskyLoss)
from pipeline.settings import (load_yaml,
                               STAT_DATA_PATH,
                               PREDICTION_PATH,
                               LOG_PATH,
                               SAMPLER_CONFIG)


class ModelSetupCallback(L.Callback):
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


class DefineCriterionCallback(L.Callback):
    def __init__(self,
                 class_path: Optional[str] = None,
                 init_args: Optional[dict] = {},
                 custom: Optional[bool] = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.class_path = class_path
        self.init_args = init_args
        self.custom = custom

    def setup(self,
              trainer: L.Trainer,
              pl_module: L.LightningModule,
              stage: str) -> None:
        # TODO: Add support for multiple criterions
        match self.class_path:
            case "BCEWithLogitsLoss":
                if self.init_args["weight"] is not None:
                    self.init_args["weight"] = torch.tensor(
                        self.init_args["weight"], device=pl_module.device)
                if self.init_args["pos_weight"] is not None:
                    self.init_args["pos_weight"] = torch.tensor(
                        self.init_args["pos_weight"], device=pl_module.device)
                criterion = torch.nn.BCEWithLogitsLoss(**self.init_args)
            case "CrossEntropyLoss":
                if self.init_args["weight"] is not None:
                    self.init_args["weight"] = torch.tensor(
                        self.init_args["weight"], device=pl_module.device)
                criterion = torch.nn.CrossEntropyLoss(**self.init_args)
            case "DiceLoss":
                if self.custom:
                    criterion = DiceLoss(**self.init_args)
                else:
                    criterion = Dice(**self.init_args)
            case "DiceBCELoss":
                criterion = DiceBCELoss(**self.init_args)
            case "FocalLoss":
                if self.custom:
                    criterion = FocalLoss(**self.init_args)
                else:
                    def criterion(i, t): return sigmoid_focal_loss(
                        i, t, **self.init_args)
            case "JacardLoss":
                if self.custom:
                    criterion = JacardLoss(**self.init_args)
                else:
                    criterion = JaccardIndex(**self.init_args)
            case "TverskyLoss":
                criterion = TverskyLoss(**self.init_args)
            case "FocalTverskyLoss":
                criterion = FocalTverskyLoss(**self.init_args)
            case "Identity":
                criterion = torch.nn.Identity()
            case _:
                return

        pl_module.criterion = criterion


class DefineActivationCallback(L.Callback):
    def __init__(self,
                 class_path: Optional[str] = None,
                 init_args: Optional[dict] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.class_path = class_path
        self.init_args = init_args

    def setup(self,
              trainer: L.Trainer,
              pl_module: L.LightningModule,
              stage: str) -> None:
        match self.class_path:
            case "Sigmoid":
                activation = torch.nn.Sigmoid()
            case "Hardsigmoid":
                activation = torch.nn.Hardsigmoid()
            case "Softmax":
                activation = torch.nn.Softmax(**self.init_args)
            case "Softmax2d":
                activation = torch.nn.Softmax2d()
            case "Softplus":
                activation = torch.nn.Softplus()
            case "Softsign":
                activation = torch.nn.Softsign()
            case "Softmin":
                activation = torch.nn.Softmin()
            case "exponential":
                activation = torch.exp
            case "ReLU":
                activation = torch.nn.ReLU()
            case "LeakyReLU":
                activation = torch.nn.LeakyReLU()
            case "ELU":
                activation = torch.nn.ELU()
            case "SELU":
                activation = torch.nn.SELU()
            case "Tanh":
                activation = torch.nn.Tanh()
            case "Identity":
                activation = torch.nn.Identity()
            case _:
                return
        pl_module.activation = activation


class LogSetupCallback(L.pytorch.cli.SaveConfigCallback):
    def save_config(self,
                    trainer: L.Trainer,
                    pl_module: L.LightningModule,
                    stage: str) -> None:
        config = copy.deepcopy(self.config)
        del config["trainer"]["logger"]
        config["config"] = [str(s) for s in config["config"]]
        config["trainer"]["callbacks"] = [{k: vars(v) if hasattr(
            v, "__dict__") else v for k, v in vars(c).items()} for c in config["trainer"]["callbacks"]]

        stat_data = load_yaml(STAT_DATA_PATH)
        if "chip_verify" in stat_data:
            del stat_data["chip_verify"]
        if "anno_verify" in stat_data:
            del stat_data["anno_verify"]
        pl_module.logger.log_hyperparams(config),
        pl_module.logger.log_hyperparams({"stat_data": stat_data,
                                          "sampler": SAMPLER_CONFIG})


class LogTrainCallback(L.Callback):
    def on_train_batch_end(self,
                           trainer: L.Trainer,
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
                                trainer: L.Trainer,
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


class LogTrainBinaryExtCallback(L.Callback):
    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: torch.Tensor,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        _, annotations, _ = batch
        classes = outputs["classes"]

        metrics = {
            "accuracy": binary_accuracy(classes, annotations),
            "jaccard_index": binary_jaccard_index(classes, annotations),
            "precision": binary_precision(classes, annotations),
            "ssim": structural_similarity_index_measure(classes, annotations),
        }

        pl_module.log_dict(
            dictionary=metrics,
            logger=True,
            prog_bar=False,
            sync_dist=True
        )


class LogTrainMulticlassExtCallback(L.Callback):
    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: torch.Tensor,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        _, annotations, _ = batch
        classes = outputs["classes"]

        metrics = {
            "accuracy": multiclass_accuracy(classes, annotations, pl_module.num_classes),
            "jaccard_index": multiclass_jaccard_index(classes, annotations, pl_module.num_classes),
            "precision": multiclass_precision(classes, annotations, pl_module.num_classes),
            "ssim": structural_similarity_index_measure(classes, annotations, pl_module.num_classes),
        }

        pl_module.log_dict(
            dictionary=metrics,
            logger=True,
            prog_bar=False,
            sync_dist=True
        )


class LogTrainPixelwiseExtCallback(L.Callback):
    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: torch.Tensor,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        _, annotations, _ = batch
        classes = outputs["classes"]

        # TODO: expand binary accuracy and precision to pixel level
        metrics = {
            "ssim": structural_similarity_index_measure(classes, annotations, pl_module.num_classes),
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
        classes = outputs["classes"]

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
            pred = classes[i]

            # save rgb and ir band separately
            rgb = chip[0:3].flip(0).permute(1, 2, 3, 0)
            ir = chip[3:6].permute(1, 2, 3, 0)
            rgb_max = torch.max(rgb).item()
            rgb_min = torch.min(rgb).item()
            ir_max = torch.max(ir).item()
            ir_min = torch.min(ir).item()
            for t in range(chip.shape[1]):
                image = rgb[t, :, :, :]
                pl_module.logger.experiment.log_image(
                    image_data=image.detach().cpu(),
                    name=f"{index:07d}_rgb_t{t}_chip",
                    image_minmax=(rgb_min, rgb_max)
                )
                image = ir[t, :, :, :]
                pl_module.logger.experiment.log_image(
                    image_data=image.detach().cpu(),
                    name=f"{index:07d}_ir_t{t}_chip",
                    image_minmax=(ir_min, ir_max)
                )

            # save original annotations
            for c in range(anno.shape[0]):
                image = anno[c].unsqueeze(-1)
                pl_module.logger.experiment.log_image(
                    image_data=image.detach().cpu(),
                    name=f"{index:07d}_c{c+1}_anno",
                    image_minmax=(0, 1)
                )

            # save predictions generated from model
            for c in range(pred.shape[0]):
                image = pred[c].unsqueeze(-1)
                pl_module.logger.experiment.log_image(
                    image_data=image.detach().cpu(),
                    name=f"{index:07d}_c{c+1}_pred",
                    image_minmax=(0, 1)
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
        classes = outputs["classes"]

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
            pred = classes[i]

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
