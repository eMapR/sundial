import lightning as L
import torch

from torch import nn

from models.base import SundialPLBase


class PrithviReshape(nn.Module):
    def __init__(self,
                view_size):
            super().__init__()
            self.view_size = view_size
    
    def forward(self, latent):
        latent = latent[:, 1:, :]
        latent = latent.view(
            latent.shape[0],
            -1,
            self.view_size,
            self.view_size)

        return latent


class PrithviBackbone(nn.Module):
    def __init__(self,
                 view_size: int,
                 prithvi_params: dict,
                 prithvi_freeze: bool = True,
                 prithvi_path: str = None,
                 reshape: bool = True):
        super().__init__()
        self.view_size = view_size
        self.prithvi_path = prithvi_path
        self.prithvi_params = prithvi_params
        self.prithvi_freeze = prithvi_freeze

        from models.backbones.prithvi.Prithvi import MaskedAutoencoderViT
        self.model = MaskedAutoencoderViT(
            **self.prithvi_params["model_args"])
        if self.prithvi_freeze:
            self.eval()
        if self.prithvi_path is not None:
            checkpoint = torch.load(self.prithvi_path)
            del checkpoint['pos_embed']
            for k, v in checkpoint.items():
                if "decoder" in k:
                    del v
            _ = self.model.load_state_dict(checkpoint, strict=False)
        if reshape:
            self.reshaper = PrithviReshape(self.view_size)
        else:
            self.reshaper = nn.Identity()

    def forward(self, chips):
        latent, _, _ = self.model.forward_encoder(chips, mask_ratio=0.0)
        return self.reshaper(latent)


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

        self.backbone = PrithviBackbone(
            view_size=view_size,
            prithvi_params=prithvi_params,
            prithvi_freeze=prithvi_freeze,
            prithvi_path=prithvi_path)

        from torchvision.models.segmentation.fcn import FCNHead
        self.head = nn.Sequential(
            Upscaler(prithvi_params["model_args"]["embed_dim"], self.upscale_depth),
            FCNHead(prithvi_params["model_args"]["embed_dim"], self.num_classes)
        )


class PrithviUNet(SundialPLBase):
    def __init__(self,
                 num_classes: int,
                 view_size: int,
                 prithvi_params: dict,
                 prithvi_freeze: bool = True,
                 prithvi_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.view_size = view_size

        self.backbone = PrithviBackbone(
            view_size=view_size,
            prithvi_params=prithvi_params,
            prithvi_freeze=prithvi_freeze,
            prithvi_path=prithvi_path)

        from models.decoders.unet import UNet
        self.head = UNet(prithvi_params["model_args"]["embed_dim"], self.num_classes)


class PrithviGlobalBackbone(nn.Module):
    def __init__(self,
                 view_size: int,
                 prithvi_params: dict,
                 prithvi_freeze: bool = True,
                 prithvi_path: str = None,
                 reshape: bool = True):
        super().__init__()
        self.view_size = view_size
        self.prithvi_path = prithvi_path
        self.prithvi_params = prithvi_params
        self.prithvi_freeze = prithvi_freeze

        from models.backbones.prithvi.PrithviGlobal import MaskedAutoencoderViT
        self.model = MaskedAutoencoderViT(
            **self.prithvi_params["model_args"])
        if self.prithvi_freeze:
            self.eval()
        if self.prithvi_path is not None:
            checkpoint = torch.load(self.prithvi_path)
            for k, v in checkpoint.items():
                if "decoder" in k:
                    del v
            _ = self.model.load_state_dict(checkpoint, strict=False)
            _ = self.model.load_state_dict(checkpoint, strict=False)
        if reshape:
            self.reshaper = PrithviReshape(self.view_size)
        else:
            self.reshaper = nn.Identity()

    def forward(self,
                chips: torch.Tensor,
                temporal_coords: torch.Tensor,
                location_coords: torch.Tensor):
        latent, _, _ = self.model.forward_encoder(chips,
                                                  temporal_coords,
                                                  location_coords,
                                                  mask_ratio=0.0)
        return self.reshaper(latent)


class PrithviGlobalFCN(SundialPLBase):
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

        self.backbone = PrithviGlobalBackbone(
            view_size=view_size,
            prithvi_params=prithvi_params,
            prithvi_freeze=prithvi_freeze,
            prithvi_path=prithvi_path)

        from torchvision.models.segmentation.fcn import FCNHead
        self.head = nn.Sequential(
            Upscaler(prithvi_params["model_args"]["embed_dim"], self.upscale_depth),
            FCNHead(prithvi_params["model_args"]["embed_dim"], self.num_classes)
        )
        
class PrithviGlobalAttentionUNet(SundialPLBase):
    def __init__(self,
        num_classes: int,
        view_size: int,
        prithvi_params: dict,
        prithvi_freeze: bool = True,
        prithvi_path: str = None,
        **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.view_size = view_size

        self.backbone = PrithviGlobalBackbone(
            view_size=view_size,
            prithvi_params=prithvi_params,
            prithvi_freeze=prithvi_freeze,
            prithvi_path=prithvi_path)

        from models.decoders.attention_unet import AttentionUNet, DownsampleBlock
        from models.decoders.utils import Upsampler
        self.head = nn.Sequential(
            Upsampler(prithvi_params["model_args"]["embed_dim"], 64),
            AttentionUNet(64, 64),
            DownsampleBlock(64, self.num_classes)
        )
