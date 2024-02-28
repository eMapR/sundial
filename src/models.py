import lightning as L
import numpy as np
import torch
import torchvision

from torch import nn


class PseudoHead(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 edge_size: int,
                 num_class_channels: int,
                 dropout: bool,
                 depth: int):
        super().__init__()
        embed_dims = [embed_dim // (4**i) for i in range(depth + 1)]

        def build_block(in_ch, out_ch, idx): return nn.Sequential(
            nn.Upsample(
                size=(edge_size//2**(depth-idx-1))),
            nn.Conv2d(
                kernel_size=3,
                in_channels=in_ch,
                out_channels=out_ch,
                padding=1),
            nn.Dropout() if dropout else nn.Identity(),
            nn.ReLU())

        self.block = nn.Sequential(
            *[build_block(embed_dims[i], embed_dims[i+1], i)
              for i in range(depth)],
            nn.Conv2d(kernel_size=1, in_channels=embed_dims[-1], out_channels=num_class_channels))

    def forward(self, x):
        return self.block(x)


class SundialPrithvi(L.LightningModule):
    def __init__(self,
                 weights_path: str,
                 num_classes: int,
                 edge_size: int,
                 learning_rate: float,
                 upscale_depth: int,
                 upscale_dropout: bool,
                 prithvi: dict):
        super().__init__()
        self.edge_size = edge_size
        self.learning_rate = learning_rate
        self.mask_ratio = prithvi["train_params"]["mask_ratio"]
        self.prithvi = prithvi

        # Initializing Prithvi Backbone per prithvi documentation
        from backbones.prithvi.Prithvi import MaskedAutoencoderViT
        checkpoint = torch.load(
            weights_path, map_location="gpu" if torch.cuda.is_available() else "cpu")
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        self.backbone = MaskedAutoencoderViT(**prithvi["model_args"])
        self.backbone.eval()
        _ = self.backbone.load_state_dict(checkpoint, strict=False)

        # Initializing Segmentation Psuedo-head
        self.head = PseudoHead(
            prithvi["model_args"]["embed_dim"]*6, edge_size, num_classes, upscale_depth, upscale_dropout)

        # Defining loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def training_step(self, batch, *args):
        image, annotations = batch
        image = image.permute(0, 4, 1, 2, 3)
        features, _, _ = self.backbone.forward_encoder(
            image, mask_ratio=self.mask_ratio)
        features = features[:, 1:, :]
        features = features.view(features.shape[0], -1, 2**4, 2**4)
        class_image = self.head(features)
        return self.criterion(class_image, annotations)

    def validation_step(self, batch, *args):
        return

    def test_step(self, batch, *args):
        return

    def predict_step(self, batch, *args):
        return

    def configure_optimizers(self, *args):
        optimizer = torch.optim.Adam(
            self.head.parameters(), lr=self.learning_rate)
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
                "name": "adam",
            }
        }
