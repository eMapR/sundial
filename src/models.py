import lightning as L
import torch

from torch import nn


class SundialPLBase(L.LightningModule):
    def forward(self, chips):
        tokens = self.backbone(chips)
        features = self.neck(tokens)
        logits = self.head(features)

        return logits

    def training_step(self, batch):
        chips, annotations, _ = batch
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        return {"loss": loss}

    def validation_step(self, batch):
        chips, annotations, _ = batch
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        # reactivating logits for metric logging
        classes = self.activation(logits)

        return {"loss": loss, "classes": classes}

    def test_step(self, batch):
        chips, annotations, _ = batch
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        # reactivating logits for metric logging
        classes = self.activation(logits)

        return {"loss": loss, "classes": classes}

    def predict_step(self, batch):
        chips, _ = batch
        logits = self(chips)
        classes = self.activation(logits)

        return {"classes": classes}


class UpscaleNeck(nn.Module):
    def __init__(self, embed_dim: int, depth: int):
        super().__init__()

        def build_block(in_ch, out_ch): return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2),)

        self.block = nn.Sequential(
            *[build_block(embed_dim, embed_dim) for _ in range(depth)])

    def forward(self, x):
        return self.block(x)


class PrithviBackbone(nn.Module):
    def __init__(self,
                 view_size: int,
                 prithvi_params: dict,
                 prithvi_freeze: bool = True,
                 prithvi_path: str = None):
        super().__init__()
        self.view_size = view_size
        self.prithvi_path = prithvi_path
        self.prithvi_params = prithvi_params
        self.prithvi_freeze = prithvi_freeze

        # Initializing Prithvi Backbone per prithvi documentation
        from backbones.prithvi.Prithvi import MaskedAutoencoderViT
        self.model = MaskedAutoencoderViT(
            **self.prithvi_params["model_args"])
        if self.prithvi_freeze:
            self.model.eval()
        if self.prithvi_path is not None:
            checkpoint = torch.load(self.prithvi_path)
            del checkpoint['pos_embed']
            del checkpoint['decoder_pos_embed']
            _ = self.model.load_state_dict(checkpoint, strict=False)

    def forward(self, chips):
        # gathering features
        tokens, _, _ = self.model.forward_encoder(chips, mask_ratio=0.0)

        # removing class token and reshaping to 2D representation
        tokens = tokens[:, 1:, :]
        tokens = tokens.view(
            tokens.shape[0],
            -1,
            self.view_size,
            self.view_size)

        return tokens


class PrithviFCN(SundialPLBase):
    def __init__(self,
                 num_classes: int,
                 upscale_depth: int,
                 view_size: int,
                 prithvi_params: dict,
                 prithvi_freeze: bool = True,
                 prithvi_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.upscale_depth = upscale_depth
        self.view_size = view_size

        # Initializing backbone
        self.backbone = PrithviBackbone(
            view_size=view_size,
            prithvi_params=prithvi_params,
            prithvi_freeze=prithvi_freeze,
            prithvi_path=prithvi_path)

        # Initializing upscaling neck
        embed_dim = prithvi_params["model_args"]["embed_dim"] * \
            prithvi_params["model_args"]["num_frames"]
        self.neck = UpscaleNeck(embed_dim, self.upscale_depth)

        # Initializing FCNHead
        from torchvision.models.segmentation.fcn import FCNHead
        self.head = FCNHead(embed_dim, self.num_classes)


class PrithviFCNDiff(PrithviFCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, chips) -> torch.Any:
        chips = chips[:, :, 1:, :, :] - chips[:, :, :-1, :, :]
        return super().forward(chips)


class PrithviUNet(SundialPLBase):
    def __init__(self,
                 num_classes: int,
                 upscale_depth: int,
                 view_size: int,
                 prithvi_params: dict,
                 prithvi_freeze: bool = True,
                 prithvi_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.upscale_depth = upscale_depth
        self.view_size = view_size

        # Initializing backbone
        self.backbone = PrithviBackbone(
            view_size=view_size,
            prithvi_params=prithvi_params,
            prithvi_freeze=prithvi_freeze,
            prithvi_path=prithvi_path)

        # Initializing upscaling neck
        embed_dim = prithvi_params["model_args"]["embed_dim"] * \
            prithvi_params["model_args"]["num_frames"]
        self.neck = UpscaleNeck(embed_dim, self.upscale_depth)

        # Initializing U-Net Head
        from heads.unet.unet.unet_model import UNet
        self.head = UNet(embed_dim, self.num_classes)


class PrithviHeadless(L.LightningModule):
    def __init__(self,
                 prithvi_params: dict,
                 **kwargs):
        super().__init__(**kwargs)
        self.prithvi_params = prithvi_params

        # initialize prithvi backbone
        from backbones.prithvi.Prithvi import MaskedAutoencoderViT
        self.model = MaskedAutoencoderViT(
            **self.prithvi_params["model_args"])

    def forward(self, chips):
        return self.model.forward(chips, mask_ratio=self.prithvi_params["train_params"]["mask_ratio"])

    def training_step(self, batch):
        chips, _ = batch
        loss, _, _ = self(chips)
        return {"loss": loss}

    def validation_step(self, batch):
        chips, _ = batch
        loss, _, _ = self(chips)
        return {"loss": loss}

    def test_step(self, batch):
        chips, _ = batch
        loss, _, _ = self(chips)
        return {"loss": loss}

    def predict_step(self, batch):
        chips, _ = batch
        features, _, _ = self.model.forward_encoder(
            chips, mask_ratio=self.prithvi_params["train_params"]["mask_ratio"])
        return {"features": features}
