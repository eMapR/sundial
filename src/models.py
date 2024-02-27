import lightning as L
import numpy as np
import torch

from torch import nn


class UpscalingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, **kwargs):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(
            kernel_size=3, in_channels=in_channels, out_channels=out_channels, padding=1)
        self.relu = nn.ReLU()
        self.block = nn.Sequential(
            *[self.upsample, self.conv, self.relu] * depth)

    def forward(self, x):
        return self.block(x)


class SundialPrithvi(L.LightningModule):
    def __init__(self,
                 weights_path: str,
                 num_classes: int,
                 embed_dim: int,
                 learning_rate: float,
                 mask_ratio: float,
                 upscale_depth: int,
                 prithvi: dict,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        print(weights_path)
        from backbones.prithvi.Prithvi import MaskedAutoencoderViT
        checkpoint = torch.load(weights_path, map_location="gpu" if torch.cuda.is_available() else "cpu")
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        self.backbone = MaskedAutoencoderViT(**prithvi["model_args"])
        self.backbone.eval()
        _ = self.backbone.load_state_dict(checkpoint, strict=False)

        self.embed_dims = [embed_dim // (2**i)
                           for i in range(upscale_depth + 1)]
        self.upscaling_block = UpscalingBlock(
            self.embed_dims[-1], self.num_classes, upscale_depth)
        self.segmentation_head = nn.Sequential(
            self.upscaling_block,
            nn.Conv2d(kernel_size=1, in_channels=self.embed_dims[-1], out_channels=self.num_classes))

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def training_step(self, batch, *args):
        image, mask = batch
        image = image.permute(0, 4, 1, 2, 3)
        features, _, _ = self.backbone.forward_encoder(
            image, mask_ratio=self.mask_ratio)

        reshaped_features = features[:, 1:, :]
        feature_img_side_length = int(np.sqrt(reshaped_features.shape[1]))
        reshaped_features = reshaped_features.view(
            -1, feature_img_side_length, feature_img_side_length, self.embed_dims["embed_dim"])
        reshaped_features = reshaped_features.permute(0, 3, 1, 2)

        return self.criterion(reshaped_features, mask)

    def validation_step(self, batch, *args):
        return

    def test_step(self, batch, *args):
        return

    def predict_step(self, batch, *args):
        return

    def configure_optimizers(self, *args):
        optimizer = torch.optim.Adam(
            self.segmentation_head.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": "adam_lr",
            }
        }
