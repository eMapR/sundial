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
        latent = latent.transpose(1,2)
        latent = latent.view(
            latent.shape[0],
            -1,
            self.view_size,
            self.view_size)

        return latent


class PrithviBackbone(nn.Module):
    def __init__(self,
                 prithvi_params: dict,
                 prithvi_freeze: bool = True,
                 prithvi_path: str = None,
                 view_size: int | None = 16,
                 reshape: bool = True,
                 decoder: bool = False):
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
            if not decoder:
                for k, v in checkpoint.items():
                    if "decoder" in k:
                        del v
            _ = self.model.load_state_dict(checkpoint, strict=False)
        self.reshaper = PrithviReshape(self.view_size) if reshape else nn.Identity()
        self.decoder = decoder
        

    def forward(self, chips):
        latent, _, ids_restore = self.model.forward_encoder(chips, mask_ratio=0.0)
        if decoder:
            latent = self.model.forward_decoder(latent, ids_restore)
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

        from models.unet import UNet
        self.head = UNet(prithvi_params["model_args"]["embed_dim"], self.num_classes)


class PrithviGlobalBackbone(nn.Module):
    def __init__(self,
                 prithvi_params: dict,
                 prithvi_freeze: bool = True,
                 prithvi_path: str = None,
                 view_size: int | None = 16,
                 reshape: bool = True,
                 decoder: bool = False):
        super().__init__()
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
            if not decoder:
                for k, v in checkpoint.items():
                    if "decoder" in k:
                        del v
            _ = self.model.load_state_dict(checkpoint, strict=False)
            
        self.reshaper = PrithviReshape(view_size) if reshape else nn.Identity()
        self.decoder = decoder

    def forward(self, data):
        if isinstance(data, dict):
            chip = data.get("chip")
            temporal = data.get("temporal_coords")
            location = data.get("location_coords")
        else:
            chip = data
            temporal = None
            location = None
        latent, _, ids_restore, = self.model.forward_encoder(chip,
                                                  temporal,
                                                  location,
                                                  mask_ratio=0.0)
        if self.decoder:
            latent = self.model.forward_decoder(latent,
                                                ids_restore,
                                                temporal,
                                                location)
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
            Upsampler(prithvi_params["model_args"]["embed_dim"], 64),
            FCNHead(256, self.num_classes)
        )
        
class PrithviGlobalAttentionUNet(SundialPLBase):
    def __init__(self,
        num_classes: int,
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

        from models.attention_unet import AttentionUNet, DownsampleBlock
        from models.utils import Upsampler
        self.head = nn.Sequential(
            Upsampler(prithvi_params["model_args"]["embed_dim"], 64),
            AttentionUNet(64, 64),
            DownsampleBlock(64, self.num_classes)
        )


class PrithviGlobalDecoder3dUNet(SundialPLBase):
    def __init__(self,
        num_classes: int,
        num_channels: int,
        kernel_size: tuple,
        prithvi_params: dict,
        prithvi_freeze: bool = True,
        prithvi_path: str = None,
        **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_size = kernel_size

        from models.utils import Conv3dBlock, ConvTranspose3dBlock
        self.down1 = Conv3dBlock(self.num_channels, 64, kernel_size=kernel_size, stride=kernel_size)
        self.down2 = Conv3dBlock(64, 128)
        self.down3 = Conv3dBlock(128, 256)
        self.down4 = Conv3dBlock(256, 512)
        self.down5 = Conv3dBlock(512, 1024) 

        self.prithvi = PrithviGlobalBackbone(
            prithvi_params=prithvi_params,
            prithvi_freeze=prithvi_freeze,
            prithvi_path=prithvi_path,
            decoder=True)

        self.up1 = Conv3dBlock(1536, 256)
        self.up2 = Conv3dBlock(512, 128)
        self.up3 = Conv3dBlock(256, 64)
        self.up4 = ConvTranspose3dBlock(128, 128, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.up5 = Conv3dBlock(128, self.num_classes)
    
    def forward(self, data):
        x = data["chip"]
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        pri_out = self.prithvi(x5)
        
        x5_concatted = torch.cat((pri_out, x4), dim=1)
        x6 = self.up1(x5_concatted)
        x6_concatted = torch.cat((x6, x3), dim=1)
        x7 = self.up2(x6_concatted)
        x7_concatted = torch.cat((x7, x2), dim=1)
        x8 = self.up3(x7_concatted)
        x8_concatted = torch.cat((x8, x1), dim=1)
        x9 = self.up4(x8_concatted)
        x10 = self.up5(x9)

        return x10
   
class PrithviGlobalDecoder2dUNet(SundialPLBase):
    def __init__(self,
        num_classes: int,
        num_channels: int,
        bilinear: bool,
        prithvi_params: dict,
        prithvi_freeze: bool = True,
        prithvi_path: str = None,
        **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.bilinear = bilinear

        from models.basic_unet import DoubleConv, Down, Up, OutConv
        self.inc = DoubleConv(self.num_channels, 256)
        self.down1 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down2 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.out = OutConv(256, self.num_classes)
        
        self.prithvi = PrithviGlobalBackbone(
            prithvi_params=prithvi_params,
            prithvi_freeze=prithvi_freeze,
            prithvi_path=prithvi_path,
            reshape=True)
        
    def forward(self, data):
        x = data["chip"]
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x4 = x3.unsqueeze(2)
        x4 = self.prithvi(x4)
        
        x5 = self.up1(x4, x2)
        x6 = self.up2(x5, x1)
        x7 = self.out(x6)
        return x7
