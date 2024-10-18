import copy
import importlib
import inspect
import lightning as L
import os
import pandas as pd
import tarfile
import torch

from torchmetrics.functional.classification import (binary_accuracy,
                                                    binary_precision,
                                                    binary_jaccard_index,
                                                    binary_recall,
                                                    multiclass_accuracy,
                                                    multiclass_precision,
                                                    multiclass_jaccard_index,
                                                    multiclass_recall)
from torchmetrics import (MeanAbsoluteError,
                          MeanSquaredError)
from torchmetrics.functional.image import structural_similarity_index_measure
from typing import Optional, Tuple

from pipeline.settings import (load_yaml,
                               EXPERIMENT_FULL_NAME,
                               LOG_PATH,
                               META_DATA_PATH,
                               PREDICTION_PATH,
                               SAMPLER_CONFIG,
                               STAT_DATA_PATH)
from utils import log_rbg_ir_image, log_tsne_plot, save_rgb_ir_tensor, tensors_to_tifs


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
                                    device=pl_module.device)
        pl_module.to_onnx(LOG_PATH, example_input)


class DefineCriterionCallback(L.Callback):
    def __init__(self,
                 class_path: Optional[str] = None,
                 init_args: Optional[dict] = {}):
        super().__init__()
        self.class_path = class_path
        self.init_args = init_args

    def setup(self,
              trainer: L.Trainer,
              pl_module: L.LightningModule,
              stage: str) -> None:
        assert self.class_path is not None

        # TODO: Add support for multiple criterions
        class_path = self.class_path.rsplit(".", 1)
        match len(class_path):
            case 1:
                modules = ["losses", "torch.nn"]
                success = False
                for module in modules:
                    try:
                        criterion_mod = importlib.import_module(module)
                        criterion_cls = getattr(
                            criterion_mod, self.class_path)
                        if "device" in inspect.signature(criterion_cls).parameters:
                            self.init_args |= {"device": pl_module.device}
                        success = True
                        break
                    except AttributeError:
                        pass
                if not success:
                    raise AttributeError(
                        f"Criterion class {self.class_path} not found in torch.nn or src.losses")
            case 2:
                module_path, class_name = class_path
                criterion_mod = importlib.import_module(module_path)
                criterion_cls = getattr(criterion_mod, class_name)
        pl_module.criterion = criterion_cls(**self.init_args)


class DefineActivationCallback(L.Callback):
    def __init__(self,
                 class_path: Optional[str] = None,
                 init_args: Optional[dict] = {}):
        super().__init__()
        self.class_path = class_path
        self.init_args = init_args

    def setup(self,
              trainer: L.Trainer,
              pl_module: L.LightningModule,
              stage: str) -> None:
        if self.class_path:
            paths = self.class_path.rsplit(".", 1)
            if len(paths) == 1:
                module = torch.nn
                class_name = self.class_path
            else:
                module_path, class_name = paths
                module = importlib.import_module(module_path)
            activation_class = getattr(module, class_name)
            activation = activation_class(**self.init_args)
            pl_module.activation = activation
        else:
            pl_module.activation = torch.nn.Identity()


class LogSetupCallback(L.pytorch.cli.SaveConfigCallback):
    def save_config(self,
                    trainer: L.Trainer,
                    pl_module: L.LightningModule,
                    stage: str) -> None:
        config = copy.deepcopy(self.config)
        del config["trainer"]["logger"]
        
        config["config"] = [str(s) for s in config["config"]]
        if config["trainer"]["callbacks"]:
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
                                batch: Tuple[torch.Tensor],
                                batch_idx: int,
                                dataloader_idx: int = 0):
        annotations = batch["anno"]
        output = outputs["output"]

        metrics = {
            "accuracy": binary_accuracy(output, annotations),
            "jaccard_index": binary_jaccard_index(output, annotations),
            "precision": binary_precision(output, annotations),
            "recall": binary_recall(output, annotations),
            "ssim": structural_similarity_index_measure(output, annotations),
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
                                batch: Tuple[torch.Tensor],
                                batch_idx: int,
                                dataloader_idx: int = 0):
        annotations = batch["anno"]
        output = outputs["output"]
        preds = torch.argmax(output, dim=1)
        target = torch.argmax(annotations, dim=1)

        metrics = {
            "accuracy": multiclass_accuracy(preds, target, pl_module.num_classes),
            "jaccard_index": multiclass_jaccard_index(preds, target, pl_module.num_classes),
            "precision": multiclass_precision(preds, target, pl_module.num_classes),
            "recall": multiclass_recall(preds, target, pl_module.num_classes),
            "ssim": structural_similarity_index_measure(preds, target, pl_module.num_classes),
        }

        pl_module.log_dict(
            dictionary=metrics,
            logger=True,
            prog_bar=False,
            sync_dist=True
        )


class LogTrainRegressionExtCallback(L.Callback):
    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: Tuple[torch.Tensor],
                                batch_idx: int,
                                dataloader_idx: int = 0):
        annotations = batch["anno"]
        output = outputs["output"]
        preds = torch.argmax(output, dim=1)
        target = torch.argmax(annotations, dim=1)

        metrics = {
            "mean_absolute_erro": MeanAbsoluteError(preds, target),
            "mean_squared_error": MeanSquaredError(preds, target),
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
                          batch: Tuple[torch.Tensor],
                          batch_idx: int,
                          dataloader_idx: int = 0):
        chips = batch["chip"]
        annotations = batch["anno"]
        indices = batch["indx"]
        loss = outputs["loss"]
        output = outputs["output"]

        pl_module.log(
            name="test_loss",
            value=loss.detach(),
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
            pred = output[i]

            # save rgb and ir band separately
            log_rbg_ir_image(chip, index, pl_module.logger.experiment)

            # save original annotations
            for c in range(anno.shape[0]):
                image = anno[c].unsqueeze(-1)
                pl_module.logger.experiment.log_image(
                    image_data=image.detach().cpu(),
                    name=f"{index:07d}_c{c+1}_anno",
                    image_scale=2.0,
                    image_minmax=(0, 1)
                )

            # save predictions generated from model
            for c in range(pred.shape[0]):
                image = pred[c].unsqueeze(-1)
                pl_module.logger.experiment.log_image(
                    image_data=image.to(torch.float32).detach().cpu(),
                    name=f"{index:07d}_c{c+1}_pred",
                    image_scale=2.0,
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
        chips = batch["chip"]
        annotations = batch["anno"]
        indices = batch["indx"]
        output = outputs["output"]

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
            pred = output[i]

            # save rgb and ir bands separately
            save_rgb_ir_tensor(chip, index, PREDICTION_PATH)

            # save annotations
            path = os.path.join(PREDICTION_PATH, f"{index:07d}_anno.pt")
            torch.save(anno, path)

            # save predictions generated from model
            path = os.path.join(PREDICTION_PATH, f"{index:07d}_pred.pt")
            torch.save(pred, path)

    def on_predict_batch_end(self,
                             trainer: L.Trainer,
                             pl_module: L.LightningModule,
                             outputs: torch.Tensor,
                             batch: Tuple[torch.Tensor],
                             batch_idx: int,
                             dataloader_idx: int = 0):
        indices = batch["indx"]
        output = outputs["output"]

        for i in range(output.shape[0]):
            index = indices[i]
            pred = output[i]
            path = os.path.join(PREDICTION_PATH, f"{index:07d}_pred.pt")
            torch.save(pred, path)


class PackageCallback(L.Callback):
    def tensor_tif_upload(self,
                          pl_module: L.LightningModule):
        tensors_dir_path = tensors_to_tifs(
            PREDICTION_PATH,
            EXPERIMENT_FULL_NAME,
            META_DATA_PATH,
            SAMPLER_CONFIG["num_workers"]
        )
        tar_path = f"{tensors_dir_path}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(tensors_dir_path, arcname=EXPERIMENT_FULL_NAME)
        pl_module.logger.experiment.log_asset(tar_path, overwrite=True)

    def on_test_end(self,
                    trainer: L.Trainer,
                    pl_module: L.LightningModule):
        self.tensor_tif_upload(pl_module)

    def on_predict_end(self,
                       trainer: L.Trainer,
                       pl_module: L.LightningModule):
        self.tensor_tif_upload(pl_module)
