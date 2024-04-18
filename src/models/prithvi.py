import lightning as L
import torch

from torch import nn

from models.base import SundialPLBase
from models.necks import UpscaleNeck


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
        from models.backbones.prithvi.Prithvi import MaskedAutoencoderViT
        self.model = MaskedAutoencoderViT(
            **self.prithvi_params["model_args"])
        if self.prithvi_freeze:
            self.eval()
        if self.prithvi_path is not None:
            checkpoint = torch.load(self.prithvi_path)
            del checkpoint['pos_embed']
            del checkpoint['decoder_pos_embed']
            _ = self.model.load_state_dict(checkpoint, strict=False)

    def forward(self, chips):
        # gathering features
        latent, _, _ = self.model.forward_encoder(chips, mask_ratio=0.0)

        # removing class token and reshaping to 2D representation
        latent = latent[:, 1:, :]
        latent = latent.view(
            latent.shape[0],
            -1,
            self.view_size,
            self.view_size)

        return latent


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
        from models.heads.unet.unet.unet_model import UNet
        self.head = UNet(embed_dim, self.num_classes)


class PrithviHeadless(L.LightningModule):
    def __init__(self,
                 view_size: int | None,
                 prithvi_params: dict,
                 **kwargs):
        super().__init__(**kwargs)
        self.view_size = view_size
        self.prithvi_params = prithvi_params

        # initialize prithvi backbone
        from models.backbones.prithvi.Prithvi import MaskedAutoencoderViT
        self.model = MaskedAutoencoderViT(
            **self.prithvi_params["model_args"])

    def forward(self, chips):
        return self.model.forward(chips, mask_ratio=self.prithvi_params["train_params"]["mask_ratio"])

    def training_step(self, batch):
        chips, _, _ = batch
        loss, _, _ = self(chips)
        return {"loss": loss}

    def validation_step(self, batch):
        chips, _, _ = batch
        loss, _, _ = self(chips)
        return {"loss": loss}

    def test_step(self, batch):
        chips, _, _ = batch
        loss, pred, _ = self(chips)

        return {"loss": loss, "pred": pred}

    def predict_step(self, batch):
        chips, _, _ = batch
        latent, _, _ = self.model.forward_encoder(chips, mask_ratio=0.0)
        return {"latent": latent}


class PrithviHeadlessCDiff(L.LightningModule):
    def __init__(self,
                 view_size: int,
                 prithvi_params: dict,
                 prithvi_freeze: bool = True,
                 prithvi_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = PrithviBackbone(
            view_size=view_size,
            prithvi_params=prithvi_params,
            prithvi_freeze=prithvi_freeze,
            prithvi_path=prithvi_path)
        assert view_size % 2 == 0
        self.center = view_size // 2
        self.center_slice = slice(self.center - 1, self.center + 1)
        self.mse = nn.MSELoss(reduction="none")

    def predict_step(self, batch):
        chips, _, _ = batch
        latent = self.model(chips)
        center = latent[:, :, self.center_slice, self.center_slice]
        center_mean = torch.mean(center, dim=(2, 3), keepdim=True)
        rmse = torch.mean(torch.sqrt(
            self.mse(latent, center_mean)), dim=1, keepdim=True)
        return {"latent": latent, "rmse": rmse}
