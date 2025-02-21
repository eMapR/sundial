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


class PrithviBackbone(L.LightningModule):
    def __init__(self,
                 prithvi_params: dict,
                 freeze_encoder: bool = True,
                 prithvi_ckpt_path: str = None,
                 view_size: int | None = 16,
                 reshape: bool = True,
                 decoder: bool = False,
                 freeze_patch_embed: bool = False):
        super().__init__()
        self.prithvi_ckpt_path = prithvi_ckpt_path
        self.prithvi_params = prithvi_params
        self.freeze_encoder = freeze_encoder

        from models.backbones.prithvi.PrithviGlobal import MaskedAutoencoderViT
        self.model = MaskedAutoencoderViT(**self.prithvi_params)
        if self.prithvi_ckpt_path is not None:
            checkpoint = torch.load(self.prithvi_ckpt_path, weights_only=False)
            if not decoder:
                for k, v in checkpoint["model"].items():
                    if "decoder" in k:
                        del v
            _ = self.model.load_state_dict(checkpoint, strict=False)
            
        self.reshaper = PrithviReshape(view_size) if reshape else nn.Identity()
        self.decoder = decoder
        if self.freeze_encoder:
            if freeze_patch_embed:
                self.model.patch_embed.eval()
                for param in self.model.patch_embed.parameters():
                    param.requires_grad = False
            self.model.blocks.eval()
            for blk in self.model.blocks:
                for param in blk.parameters():
                    param.requires_grad = False

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


class PrithviFCN(SundialPLBase):
    def __init__(self,
                 num_classes: int,
                 upscale_depth: int,
                 view_size: int,
                 prithvi_params: dict,
                 freeze_encoder: bool = True,
                 prithvi_ckpt_path: str = None,
                 **kwargs):
        super()._init__(**kwargs)
        self.num_classes = num_classes
        self.upscale_depth = upscale_depth
        self.view_size = view_size

        self.backbone = PrithviBackbone(
            view_size=view_size,
            prithvi_params=prithvi_params,
            freeze_encoder=freeze_encoder,
            prithvi_ckpt_path=prithvi_ckpt_path)

        from torchvision.models.segmentation.fcn import FCNHead
        from models.utils import Upsampler
        self.head = nn.Sequential(
            Upsampler(prithvi_params["model_args"]["embed_dim"], 64),
            FCNHead(256, self.num_classes)
        )


class PrithviDecoder3dUNet(SundialPLBase):
    def __init__(self,
        num_classes: int,
        num_channels: int,
        num_frames: int,
        kernel_size: tuple | int,
        stride: tuple | int,
        padding: tuple | int,
        prithvi_params: dict,
        freeze_encoder: bool = True,
        prithvi_ckpt_path: str = None,
        freeze_patch_embed: bool = False,
        **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_frames = num_frames
        params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }
        
        from models.utils import DoubleConv3d, Up3d, OutConv3d
        self.inc = DoubleConv3d(self.num_channels, 64, **params)
        self.down1 = DoubleConv3d(64, 128, **params)
        self.down2 = DoubleConv3d(128, 256, **params)
        self.down3 = DoubleConv3d(256, 512, **params)
        self.down4 = DoubleConv3d(512, 1024, **params)

        self.prithvi = PrithviBackbone(
            prithvi_params=prithvi_params,
            freeze_encoder=freeze_encoder,
            prithvi_ckpt_path=prithvi_ckpt_path,
            decoder=True,
            reshape=False,
            freeze_patch_embed=freeze_patch_embed)

        # self.prithvi = DoubleConv3d(1024, 1024, **params) # leftover from ablation

        self.up1 = Up3d(1024, 512, **params)
        self.up2 = Up3d(512, 256, **params)
        self.up3 = Up3d(256, 128, **params)
        self.up4 = Up3d(128, 64, **params)
        params["kernel_size"] = (self.num_frames, 3, 3)
        self.out = OutConv3d(64, self.num_classes, **params)
    
    def forward(self, data):
        x1 = data["chip"]
        x1 = self.inc(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.prithvi(x5)
        x6 = self.prithvi.model.unpatchify(x6)
        
        x7 = self.up1(x6, x4)
        x8 = self.up2(x7, x3)
        x9 = self.up3(x8, x2)
        x10 = self.up4(x9, x1)
        x11 = self.out(x10).squeeze(2)
        return x11


class Prithvi3dUNet(SundialPLBase):
    def __init__(self,
        num_classes: int,
        num_channels: int,
        num_frames: int,
        kernel_size: tuple | int,
        stride: tuple | int,
        padding: tuple | int,
        prithvi_params: dict,
        freeze_encoder: bool = True,
        prithvi_ckpt_path: str = None,
        freeze_patch_embed: bool = False,
        **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_frames = num_frames

        from models.utils import DoubleConv3d
        from models.unet2d import Up, OutConv
        params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }
        
        self.inc = DoubleConv3d(self.num_channels, 64, **params)
        self.down1 = DoubleConv3d(64, 128, **params)
        self.down2 = DoubleConv3d(128, 256, **params)
        self.down3 = DoubleConv3d(256, 512, **params)
        self.down4 = DoubleConv3d(512, 1024, **params)

        self.prithvi = PrithviBackbone(
            prithvi_params=prithvi_params,
            freeze_encoder=freeze_encoder,
            prithvi_ckpt_path=prithvi_ckpt_path,
            decoder=False,
            reshape=True,
            freeze_patch_embed=freeze_patch_embed)

        self.up1 = Up(1024, 512, **params)
        self.up2 = Up(512, 256, **params)
        self.up3 = Up(256, 128, **params)
        self.up4 = Up(128, 64, **params)
        self.out = OutConv(64, self.num_classes, **params)
    
    def forward(self, data):
        x = data["chip"]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.prithvi(x5)
        
        x7 = self.up1(x6, x4)
        x8 = self.up2(x7, x3)
        x9 = self.up3(x8, x2)
        x10 = self.up4(x9, x1)
        x11 = self.out(x10)
        return x11


class PrithviDecoder2dUNet(SundialPLBase):
    def __init__(self,
        num_classes: int,
        num_channels: int,
        kernel_size: tuple,
        patch_size: tuple,
        padding: tuple,
        prithvi_params: dict,
        freeze_encoder: bool = True,
        prithvi_ckpt_path: str = None,
        freeze_patch_embed: bool = False,
        **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_channels = num_channels
        

        from models.unet2d import DoubleConv2d, Up2d, Down2d, OutConv2d
        params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }
        
        self.inc = DoubleConv2d(self.num_channels, 64, **params)
        self.down1 = Down2d(64, 128, **params)
        self.down2 = Down2d(128, 256, **params)
        self.down3 = Down2d(256, 512, **params)
        self.down4 = Down2d(512, 1024, **params)

        self.up1 = Up2d(1024, 512, **params)
        self.up2 = Up2d(512, 256, **params)
        self.up3 = Up2d(256, 128, **params)
        self.up4 = Up2d(128, 64, **params)
        self.out = OutConv2d(64, self.num_classes)
        
        self.prithvi = PrithviBackbone(
            prithvi_params=prithvi_params,
            freeze_encoder=freeze_encoder,
            prithvi_ckpt_path=prithvi_ckpt_path,
            decoder=True,
            reshape=False,
            freeze_patch_embed=freeze_patch_embed)
        
    def forward(self, data):
        x = data["chip"]
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x6 = x4.unsqueeze(2)
        x6 = self.prithvi(x4)
        x6 = self.prithvi.model.unpatchify(x4)
        x6 = x4.squeeze(2)
        
        x7 = self.up1(x6, x4)
        x8 = self.up2(x7, x3)
        x9 = self.up3(x8, x2)
        x10 = self.up4(x9, x1)
        x11 = self.out(x10)
        return x11
    

class Prithvi2dUNet(SundialPLBase):
    def __init__(self,
        num_classes: int,
        num_channels: int,
        prithvi_params: dict,
        freeze_encoder: bool = True,
        prithvi_ckpt_path: str = None,
        freeze_patch_embed: bool = False,
        **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.bilinear = bilinear

        from models.unet2d import DoubleConv2d, Up2d, Down2d, OutConv2d
        self.inc = DoubleConv2d(self.num_channels, 64)
        self.down1 = Down2d(64, 128)
        self.down2 = Down2d(128, 256)
        self.down3 = Down2d(256, 512)
        self.down4 = Down2d(512, 1024)

        self.up1 = Up2d(1024, 512)
        self.up2 = Up2d(512, 256)
        self.up3 = Up2d(256, 128)
        self.up4 = Up2d(128, 64)
        self.out = OutConv2d(64, self.num_classes)
        
        self.prithvi = PrithviBackbone(
            prithvi_params=prithvi_params,
            freeze_encoder=freeze_encoder,
            prithvi_ckpt_path=prithvi_ckpt_path,
            decoder=False,
            reshape=True,
            freeze_patch_embed=freeze_patch_embed)
        
    def forward(self, data):
        x = data["chip"]
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = x4.unsqueeze(2)
        x6 = self.prithvi(x5)
        
        x7 = self.up1(x6, x5)
        x8 = self.up2(x7, x4)
        x9 = self.up3(x8, x3) 
        x10 = self.up4(x9, x2)
        x11 = self.out(x10)
        
        return x11
    
class PrithviEmbed(SundialPLBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.prithvi = PrithviBackbone(**kwargs)
        
    def forward(self, batch):
        return self.prithvi(batch["chip"])
        
    def predict_step(self, batch):
        output = self(batch)
        return {"output": output.detach(), "anno": batch["anno"]} 
    

class PrithviDiffReconstruction(SundialPLBase):
    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 freeze_encoder: bool = True):
        super().__init__()
        self.prithvi = PrithviBackbone(
            prithvi_params=prithvi_params,
            freeze_encoder=freeze_encoder,
            prithvi_ckpt_path=prithvi_ckpt_path,
            decoder=True,
            reshape=False)
        input_size = prithvi_params["model_args"]["input_size"]
        patch_size = prithvi_params["model_args"]["patch_size"]
        s, p, q = patch_size
        s -= 1
        D = s * p * q * prithvi_params["model_args"]["in_chans"]
        gs = [s // p for s, p in zip(input_size, patch_size)]
        self.L = gs[1] * gs[2]
        
        self.prithvi.model.decoder_pred = nn.Linear(512,
                                      D,
                                      bias=True)

        from einops import rearrange
        self.prithvi.model.patchify = lambda imgs: rearrange(imgs, 'b c (t s) (h p) (w q) -> b (t h w) (s p q c)', s=s, p=p, q=q)
        self.prithvi.model.unpatchify = lambda imgs: rearrange(imgs, 'b (t h w) (s p q c) -> b c (t s) (h p) (w q)', h=gs[1], w=gs[2], t=gs[0], s=s, p=p, q=q)

        from models.utils import FirstOrderDifference
        self.fo_diff = FirstOrderDifference()

    def forward(self, data):
        return self.prithvi(data)
    
    def training_step(self, batch):
        data = batch
        diff = self.fo_diff(data["chip"])
        pred = self(data)
        
        mask = torch.ones([data["chip"].shape[0], self.L], device=diff.device)
        loss = self.prithvi.model.forward_loss(diff, pred, mask)
        # recon = self.prithvi.model.unpatchify(pred) # boy this will slow down training...
        
        return {"loss": loss}

    def validation_step(self, batch):
        data = batch
        diff = self.fo_diff(data["chip"])
        pred = self(data)
        
        mask = torch.ones([data["chip"].shape[0], self.L], device=diff.device)
        loss = self.prithvi.model.forward_loss(diff, pred, mask)
        recon = self.prithvi.model.unpatchify(pred)

        return {"loss": loss, "output": recon, "diff": diff}
    
    def test_step(self, batch):
        data = batch
        diff = self.fo_diff(data["chip"])
        pred = self(data)
        
        mask = torch.ones([data["chip"].shape[0], self.L], device=diff.device)
        loss = self.prithvi.model.forward_loss(diff, pred, mask)
        recon = self.prithvi.model.unpatchify(pred)

        return {"loss": loss, "output": recon, "diff": diff}
