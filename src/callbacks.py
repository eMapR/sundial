import copy
import importlib
import inspect
import lightning as L
import numpy as np
import os
import pandas as pd
import tarfile
import torch
import xarray as xr

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
from torchvision.transforms.functional import to_pil_image
from typing import Optional, Tuple

from config_utils import load_yaml
from constants import (DATA_PATH,
                       EXPERIMENT_FULL_NAME,
                       IDX_NAME_ZFILL,
                       LOG_PATH,
                       PREDICTIONS_PATH,
                       STAT_DATA_PATH)
from pipeline.settings import PIPELINE_CONFIG
from utils import dummy_zarr, log_rgb_image, log_false_color_image, save_rgb_ir_tensor, tensors_to_tifs


class ModelSetupCallback(L.Callback):
    def setup(self,
              trainer: L.Trainer,
              pl_module: L.LightningModule,
              stage: str):
        pl_module.strict_loading = False


class GenerateTrainGifCallback(L.Callback):
    def __init__(self,
                 index_name: str):
        super().__init__()
        self.index_name = index_name
        self.index = int(self.index_name)
        self.input = False
        self.frames = []
    
    def on_train_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: Tuple[torch.Tensor],
                                batch_idx: int,
                                dataloader_idx: int = 0):
        if self.index in batch["indx"]:
            batch_index = (batch["indx"] == self.index).nonzero(as_tuple=True)[0].item()
            frame_tensor = outputs["output"][batch_index]
            self.frames.append(frame_tensor)
            if not self.input:
                anno = batch["anno"][batch_index]
                chip = batch["chip"][batch_index]
                log_rgb_image(chip, f"{self.index_name}_gif", "chip", pl_module.logger.experiment)
                for c in range(anno.shape[0]):
                    pl_module.logger.experiment.log_image(
                        image_data=anno[c].unsqueeze(-1),
                        name=f"{self.index_name}_c{c+1}_gif_anno_gif.png",
                        image_scale=2.0,
                        image_minmax=(0, 1)
                    )
                self.input = True
        
    def on_train_end(self,
                    trainer: L.Trainer,
                    pl_module: L.LightningModule):
        for c in range(self.frames[0].shape[0]):
            pil_frames = [to_pil_image(frame[c]) for frame in self.frames]
            if pil_frames:
                pil_frames[0].save(PREDICTIONS_PATH + f"/{self.index_name}_out_per_epoch_c{c}.gif", save_all=True, append_images=pil_frames[1:], duration=200, loop=0)


class GenerateValidateGifCallback(L.Callback):
    def __init__(self,
                 index_name: str):
        super().__init__()
        self.index_name = index_name
        self.index = int(self.index_name)
        self.input = False
        self.frames = []
    
    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: Tuple[torch.Tensor],
                                batch_idx: int,
                                dataloader_idx: int = 0):
        outputs = outputs
        batch = batch
        
        if self.index in batch["indx"]:
            batch_index = (batch["indx"] == self.index).nonzero(as_tuple=True)[0].item()
            frame_tensor = outputs["output"][batch_index]
            self.frames.append(frame_tensor)
            if not self.input:
                anno = batch["anno"][batch_index]
                chip = batch["chip"][batch_index]
                log_rgb_image(chip, f"{self.index_name}_gif", "chip", pl_module.logger.experiment)
                for c in range(anno.shape[0]):
                    pl_module.logger.experiment.log_image(
                        image_data=anno[c].unsqueeze(-1),
                        name=f"{self.index_name}_c{c+1}_gif_anno.png",
                        image_scale=2.0,
                        image_minmax=(0, 1)
                    )
                self.input = True       
        
    def on_train_end(self,
                    trainer: L.Trainer,
                    pl_module: L.LightningModule):
        for c in range(self.frames[0].shape[0]):
            pil_frames = [to_pil_image(frame[c]) for frame in self.frames]
            if pil_frames:
                pil_frames[0].save(PREDICTIONS_PATH + f"/{self.index_name}_out_per_epoch_c{c}.gif", save_all=True, append_images=pil_frames[1:], duration=200, loop=0)

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
        for k, v in outputs.items():
            if k != "loss":
                pl_module.log(
                    name=k,
                    value=v,
                    logger=True,
                    prog_bar=False,
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


class LogTrainImageCallback(L.Callback):
    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: Tuple[torch.Tensor],
                                batch_idx: int,
                                dataloader_idx: int = 0):
        diff = outputs["diff"]
        output = outputs["output"]

        ssim = sum([structural_similarity_index_measure(output[:,:,i,...], diff[:,:,i,...])  for i in output.shape[2]]) / output.shape[2]
        
        metrics = {
            "ssim": ssim,
        }

        pl_module.log_dict(
            dictionary=metrics,
            logger=True,
            prog_bar=False,
            sync_dist=True
        )


class LogBinaryExtCallback(L.Callback):
    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: torch.Tensor,
                                batch: Tuple[torch.Tensor],
                                batch_idx: int,
                                dataloader_idx: int = 0):
        self.calc_and_log(pl_module, outputs, batch)
    
    def on_test_batch_end(self,
                        trainer: L.Trainer,
                        pl_module: L.LightningModule,
                        outputs: torch.Tensor,
                        batch: Tuple[torch.Tensor],
                        batch_idx: int,
                        dataloader_idx: int = 0):
        self.calc_and_log(pl_module, outputs, batch)
        
        
    def calc_and_log(self,
                     pl_module: L.LightningModule,
                     outputs: torch.Tensor,
                     batch: Tuple[torch.Tensor]):
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
        
        preds = torch.argmax(output, dim=1, keepdim=True).to(torch.float32)
        target = torch.argmax(annotations, dim=1, keepdim=True).to(torch.float32)

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
            "mean_absolute_error": MeanAbsoluteError(preds, target),
            "mean_squared_error": MeanSquaredError(preds, target),
        }

        pl_module.log_dict(
            dictionary=metrics,
            logger=True,
            prog_bar=False,
            sync_dist=True
        )

        
class LogAvgMagGradientCallback(L.Callback):
    def __init__(self,
                 layers: list[str],
                 freq: int = 16,
                 **kwargs):
        super().__init__(**kwargs)
        self.layers = layers
        self.freq = freq

    def on_after_backward(self,
                          trainer: L.Trainer,
                          pl_module: L.LightningModule):
        if trainer.global_step % self.freq == 0:
            mags = {f"{k}_avgmag": [] for k in self.layers}

            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    mag = torch.abs(param.grad.clone().norm()).item() 
                    for layer in self.layers:
                        if layer in name:
                            mags[f"{layer}_avgmag"].append(mag)
                            continue
            
            log_dict = {}
            for k, v in mags.items():
                if count := len(mags[k]) > 0:
                    log_dict[k] = sum(mags[k]) / count

            pl_module.log_dict(
                dictionary=log_dict,
                logger=True,
                prog_bar=False,
                sync_dist=True
            )
            

class DelayedModelCheckpoint(L.pytorch.callbacks.ModelCheckpoint):
    def __init__(self, start_after_step:int =0, start_after_epoch: int=0, **kwargs):
        super().__init__(**kwargs)
        self.start_after_step = start_after_step
        self.start_after_epoch = start_after_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_after_epoch:
            super().on_validation_end(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step >= self.start_after_step:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        
class UpdateDataSamplerExtCallback(L.Callback):
    def on_train_batch_end(self,
                           trainer: L.Trainer,
                           pl_module: L.LightningModule,
                           outputs,
                           batch,
                           batch_idx):
        pl_module.update_ema()    
    
    def on_train_epoch_end(self,
                           trainer: L.Trainer,
                           pl_module: L.LightningModule):
        trainer.train_dataloader.dataset._sampler.resample()
        
        
class SavePixelPatchCallback(L.Callback):
    def __init__(self, outpath: str, **kwargs):
        super().__init__(**kwargs)
        self.outpath = outpath
        
    def on_test_batch_end(self,
                          trainer: L.Trainer,
                          pl_module: L.LightningModule,
                          outputs,
                          batch,
                          batch_idx):
        B = int(outputs.shape[0])
        y, x = batch["meta"]["point"]

        for b in range(B):
            name = f"{y[b]}_{x[b]}.pt"
            torch.save(outputs[b], os.path.join(self.outpath, name))
            
class SaveLinesCallback(L.Callback):
    def __init__(self, dtype, chunk_sizes, shape, dims, out_path, **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype
        self.chunk_sizes = chunk_sizes
        self.dims = dims
        self.outpath = out_path
        
    def on_test_start(self, trainer, pl_module):
        imagery_da = trainer.train_dataloader.dataset._imagery_da
        sample_size = trainer.train_dataloader.dataset._sample_size
        y_coords = np.meshgrid(imagery_da.coords["lat"][-_sample_size:])
        x_coords = np.meshgrid(imagery_da.coords["lon"][-_sample_size:])
        dummy_zarr(self.dtype, self.chunk_sizes, self.shape, self.dims, y_coords, x_coords, self.out_path)
        
    def on_test_batch_end(self,
                          trainer: L.Trainer,
                          pl_module: L.LightningModule,
                          outputs,
                          batch,
                          batch_idx):
        lats, lons = batch["meta"]["point"]
        C = self.chunk_sizes[0]
        T = self.chunk_sizes[1]
        array = output.permute(1, 2, 0).reshape(C, T, 1, -1)
        chunk_da = xr.DataArray(
            array,
            dims=["band", "time", "lat", "lon"],
            coords={
                "band": list(range(C)),
                "time": list(range(T)),
                "lat": lats,
                "lon": lons
            },
            name="dat"
        )
        
        lat_start = np.searchsorted(-self._lat_coords, -lats[0])
        lon_start = np.searchsorted(self._lon_coords, lons[0])

        region = {
            "band": slice(0, self._chunk_sizes[0]),
            "time": slice(0, self._chunk_sizes[1]),
            "lat": slice(lat_start, lat_start + len(lats)),
            "lon": slice(lon_start, lon_start + len(lons)),
        }

        chunk_da.to_zarr(self._source_path, region=region)
        