import lightning as L
import torch

from torch import nn


class UpscaleNeck(nn.Module):
    def __init__(self, embed_dims: int):
        super().__init__()

        def build_block(in_ch, out_ch): return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2),
            nn.ReLU())

        self.block = nn.Sequential(
            *[build_block(embed_dims[i], embed_dims[i+1])
              for i in range(len(embed_dims) - 1)])

    def forward(self, x):
        return self.block(x)


class FCNHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      inter_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels,
                      out_channels,
                      kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class SundialPrithvi(L.LightningModule):
    def __init__(self,
                 num_classes: int,
                 view_size: int,
                 upscale_depth: int,
                 upscale_reduction_factor: int,
                 prithvi_path: str,
                 prithvi_params: dict):
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.view_size = view_size
        self.mask_ratio = prithvi_params["train_params"]["mask_ratio"]

        # Initializing Prithvi Backbone per prithvi documentation
        from backbones.prithvi.Prithvi import MaskedAutoencoderViT
        checkpoint = torch.load(
            prithvi_path, map_location="gpu" if torch.cuda.is_available() else "cpu")
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        self.backbone = MaskedAutoencoderViT(**prithvi_params["model_args"])
        self.backbone.eval()
        _ = self.backbone.load_state_dict(checkpoint, strict=False)

        # Initializing upscaling neck
        embed_dim = prithvi_params["model_args"]["embed_dim"] * \
            prithvi_params["model_args"]["num_frames"]
        embed_dims = [embed_dim // (upscale_reduction_factor**i)
                      for i in range(upscale_depth + 1)]
        self.neck = UpscaleNeck(embed_dims)

        # Initializing FCNHead
        self.head = FCNHead(embed_dims[-1], num_classes)

        # Defining loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x):
        # gathering features
        features, _, _ = self.backbone.forward_encoder(
            x, mask_ratio=self.mask_ratio)

        # removeing class token and reshaping to 2D representation
        features = features[:, 1:, :]
        features = features.view(
            features.shape[0], -1, self.view_size, self.view_size)

        # performating segmentation
        features = self.neck(features)
        image = self.head(features)

        return image

    def training_step(self, batch, batch_idx):
        # reshaping gee data (N D H W C) to pytorch format (N C D H W)
        image, annotations = batch
        image = image.permute(0, 1, 4, 2, 3)

        # performating segmentation
        class_image = self.forward(image)

        # calculating loss
        loss = self.criterion(class_image, annotations)

        # logging loss
        self.log(
            name="train/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # reshaping gee data (N D H W C) to pytorch format (N C D H W)
        image, annotations = batch
        image = image.permute(0, 1, 4, 2, 3)

        # performating segmentation
        class_image = self.forward(image)

        # calculating loss
        loss = self.criterion(class_image, annotations)

        self.log(
            name="val/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        # reshaping gee data (N D H W C) to pytorch format (N C D H W)
        image, annotations = batch
        image = image.permute(0, 1, 4, 2, 3)

        # performating segmentation
        class_image = self.forward(image)

        # calculating loss
        loss = self.criterion(class_image, annotations)

        # logging loss
        self.log(
            name="test/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def predict_step(self, batch, batch_idx):
        # reshaping gee data (N D H W C) to pytorch format (N C D H W)
        image, _ = batch
        image = image.permute(0, 1, 4, 2, 3)
        class_image = self.forward(image)
        return class_image
