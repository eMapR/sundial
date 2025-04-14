import gc
import lightning as L
import torch
import torch.nn.functional as F

from torch import nn

from models.base import SundialPLBase
from models.utils import InteractionBlock, SpatialPriorModule, deform_inputs


class PrithviReshape(nn.Module):
    def __init__(self,
                patch_size,
                input_size):
            super().__init__()
            self.patch_size = patch_size
            self.input_size = input_size
            self.view_size = self.input_size // self.patch_size[-1]
    
    def forward(self, latent):
        latent = latent[:, 1:, :]
        latent = latent.transpose(1,2)
        latent = latent.reshape(
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
                 reshape: bool = True):
        super().__init__()
        self.prithvi_ckpt_path = prithvi_ckpt_path
        self.prithvi_params = prithvi_params
        self.freeze_encoder = freeze_encoder

        from models.backbones.prithvi.prithvi2 import PrithviMAE
        self.model = PrithviMAE(**self.prithvi_params)
        if self.prithvi_ckpt_path is not None:
            checkpoint = torch.load(self.prithvi_ckpt_path, weights_only=False)
            if "encoder.pos_embed" not in checkpoint.keys():
                key = "model" if "model" in checkpoint.keys() else "state_dict"
                keys = list(checkpoint[key].keys())
                checkpoint = checkpoint[key]
            else:
                keys = list(checkpoint.keys())
            for k in keys:
                if ((prithvi_params["encoder_only"]) and ("decoder" in k)) or "pos_embed" in k:
                    del checkpoint[k]
                elif "prithvi" in k:
                    print(f"Warning: renaming prithvi layer {k}")
                    new_k = k.replace("prithvi.", "")
                    checkpoint[new_k] = checkpoint[k]
                elif k in self.model.state_dict() and checkpoint[k].shape != self.model.state_dict()[k].shape:
                    print(f"Warning: size mismatch for layer {k}, deleting: {checkpoint[k].shape} != {self.model.state_dict()[k].shape}")
                    del checkpoint[k]
                
            _ = self.model.load_state_dict(checkpoint, strict=False)
            
        self.reshaper = PrithviReshape(prithvi_params["patch_size"], prithvi_params["img_size"]) if reshape else nn.Identity()
        if self.freeze_encoder:
            self.model.encoder.eval()
            for blk in self.model.encoder.blocks:
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
        
        if self.prithvi_params["encoder_only"]:
            latent = self.model.forward_features(chip,
                                                 temporal,
                                                 location)
        else:
            latent, mask, ids_restore = self.model.encoder(chip, temporal, location, 0.0)
            latent = self.model.decoder(latent,
                                ids_restore,
                                temporal,
                                location,
                                input_size=(self.prithvi_params["num_frames"], self.prithvi_params["img_size"], self.prithvi_params["img_size"]))
        return self.reshaper(latent)


class PrithviBackboneOnly(PrithviBackbone, SundialPLBase):
    pass


class PrithviFCN(SundialPLBase):
    def __init__(self,
                 num_classes: int,
                 prithvi_params: dict,
                 freeze_encoder: bool = True,
                 prithvi_ckpt_path: str = None,
                 ablate: bool = False,
                 embed: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.ablate = ablate
        self.embed = embed
        
        if self.ablate:
            self.prithvi =  DoubleConv2d(prithvi_params["in_chans"], 1024),
        else:
            self.prithvi = PrithviBackbone(
                prithvi_params=prithvi_params,
                freeze_encoder=freeze_encoder,
                prithvi_ckpt_path=prithvi_ckpt_path,
                reshape=not embed)
        
        if not self.embed:
            from torchvision.models.segmentation.fcn import FCNHead
            from models.utils import Upscaler
            self.head = nn.Sequential(
                Upscaler(prithvi_params["embed_dim"]*prithvi_params["num_frames"] , 4),
                FCNHead(128, self.num_classes)
            )
            
        
    def forward(self, data):
        if self.ablate:
            B, C, T, H, W = data["chip"].shape
            x = data["chip"].reshape(B, -1, H, W)
            latent = self.prithvi(x)
        else:
            latent = self.prithvi(data)
        if self.embed:
            return latent
        else:
            predictions = self.head(latent)
            return predictions


class PrithviDecoder3dUNet(SundialPLBase):
    def __init__(self,
        num_classes: int,
        num_channels: int,
        kernel_size: tuple | int,
        stride: tuple | int,
        padding: tuple | int,
        prithvi_params: dict,
        freeze_encoder: bool = True,
        prithvi_ckpt_path: str = None,
        ablate: bool = False,
        embed: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.ablate = ablate
        self.embed = embed
        params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }
        
        from models.utils import DoubleConv3d, Up3d, OutConv3d
        self.inc = DoubleConv3d(self.num_channels, 16, **params)
        self.down1 = DoubleConv3d(16, 32, **params)
        self.down2 = DoubleConv3d(32, 64, **params)
        self.down3 = DoubleConv3d(64, 128, **params)

        if ablate:
            self.prithvi = DoubleConv3d(128, 128, **params)
        else:
            self.prithvi = PrithviBackbone(
                prithvi_params=prithvi_params,
                freeze_encoder=freeze_encoder,
                prithvi_ckpt_path=prithvi_ckpt_path,
                reshape=False)

        self.up1 = Up3d(256, 128, **params)
        self.up2 = Up3d(192, 64, **params)
        self.up3 = Up3d(96, 32, **params)
        self.up4 = Up3d(48, 16, **params)
        params["kernel_size"] = (prithvi_params["num_frames"], 3, 3)
        params["padding"] = (0, 1, 1)
        self.out = OutConv3d(16, self.num_classes, **params)
    
    def forward(self, data):
        x1 = data["chip"]
        x1 = self.inc(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x5 = self.prithvi(x4)
        if self.embed:
            return x5
        if not self.ablate:
            x5 = self.prithvi.model.unpatchify(x5)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.out(x9).squeeze(2)
        return x10


class PrithviAdapter(SundialPLBase):
    """
    Prithvi HLS model with ViT-adapter.

    Based on https://github.com/czczup/ViT-Adapter
    """

    def __init__(
        self,
        num_classes: int,
        prithvi_params: dict,
        freeze_encoder: bool = True,
        prithvi_ckpt_path: str = None,
        embed: bool = False,
        interaction_num_heads: int = 8,
        interaction_indexes: list[list[int]] = [[0, 1], [1, 2], [2, 3]],
        drop_channels_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.interaction_indexes = interaction_indexes
        self.D = prithvi_params["embed_dim"]*prithvi_params["num_frames"]

        self.level_embed = nn.Parameter(data=torch.zeros(3, self.D))
        self.spm = SpatialPriorModule(
            in_channels=prithvi_params["in_chans"], inplanes=64, embed_dim=prithvi_params["embed_dim"], num_frames=prithvi_params["num_frames"]
        )
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=prithvi_params["embed_dim"],
                    num_frames=prithvi_params["num_frames"],
                    num_heads=interaction_num_heads,  # embed_dims must be divisible by num_heads
                    with_cffn=False,
                    extra_extractor=(  # use_extra_extractor
                        True if i == len(interaction_indexes) - 1 else False
                    ),
                    **kwargs,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(self.D, self.D, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.D)
        self.norm2 = nn.SyncBatchNorm(self.D)
        self.norm3 = nn.SyncBatchNorm(self.D)
        self.norm4 = nn.SyncBatchNorm(self.D)

        self.backbone: torch.nn.Module = PrithviBackbone(
                prithvi_params=prithvi_params,
                freeze_encoder=freeze_encoder,
                prithvi_ckpt_path=prithvi_ckpt_path,
                reshape=False)
        self.drop_channels = nn.Dropout2d(drop_channels_rate) if drop_channels_rate > 0 else nn.Identity()
        
        from models.utils import Upscaler
        self.upscaler = Upscaler(prithvi_params["embed_dim"]*prithvi_params["num_frames"], 2)
        self.out_conv = nn.Conv2d(int(self.D//(2**2)), num_classes, kernel_size=1)
        
        self.Ps = prithvi_params["patch_size"][-1]
        self.P = prithvi_params["img_size"] // self.Ps
        
    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input["chip"]
        B, C, T, H, W = input.shape

        # deform_inputsN = [reference_points, spatial_shapes, level_start_index]
        deform_inputs1, deform_inputs2 = deform_inputs(x=input, ps=self.Ps)

        # Spatial Prior Module (SPM) forward
        c1, c2, c3, c4 = self.spm(input)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x = self.backbone.model.encoder.patch_embed(input)
        _, L, D = x.shape
        pos_embed = self.backbone.model.encoder.pos_embed[:, 1:]
        x = self.drop_channels(x + pos_embed)
        
        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(  # pass into InteractionBlock
                x=x,
                c=c,
                blocks=self.backbone.model.encoder.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs1=deform_inputs1,
                deform_inputs2=deform_inputs2,
                H=H,
                W=W,
            )

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c1 = c1.transpose(1, 2).view(B, D*T, 56, 56).contiguous()
        c2 = c2.transpose(1, 2).view(B, D*T, 28, 28).contiguous()  # HW*2
        c3 = c3.transpose(1, 2).view(B, D*T, 14, 14).contiguous()  # HW
        c4 = c4.transpose(1, 2).view(B, D*T, 7, 7).contiguous()  # HW/2

        c1 = self.up(c2) + c1

        x = x.reshape(B, T, int(L//T), D).permute(0, 2, 1, 3).reshape(B, int(L//T), D*T)
        x3 = x.transpose(1, 2).view(B, D*T, self.P, self.P).contiguous()
        x1 = F.interpolate(x3, scale_factor=4, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.interpolate(x3, scale_factor=0.5, mode="bilinear", align_corners=False)
    
        c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        
        f4_up = F.interpolate(f4, scale_factor=2, mode="bilinear", align_corners=False)
        f3_fused = f3 + f4_up

        f3_up = F.interpolate(f3_fused, scale_factor=2, mode="bilinear", align_corners=False)
        f2_fused = f2 + f3_up

        f2_up = F.interpolate(f2_fused, scale_factor=2, mode="bilinear", align_corners=False)
        f1_fused = f1 + f2_up
        
        up_scale = self.upscaler(f1_fused)
        out = self.out_conv(up_scale)
        return out


class PrithviMosaicEmbedding(SundialPLBase):
    def __init__(self,
                 prithvi_params: dict,
                 freeze_encoder: bool = True,
                 prithvi_ckpt_path: str = None,
                 stride: int = 1,
                 ablate: bool = False):
        super().__init__()
        self.stride = stride
        self.ablate = ablate
        
        if self.ablate:
            self.prithvi =  DoubleConv2d(prithvi_params["in_chans"], 1024),
        else:
            self.prithvi = PrithviBackbone(
                prithvi_params=prithvi_params,
                freeze_encoder=freeze_encoder,
                prithvi_ckpt_path=prithvi_ckpt_path,
                reshape=True)
        
        import torchvision.transforms.v2 as T
        self.kernel_size = (prithvi_params["img_size"], prithvi_params["img_size"])
        self.D = prithvi_params["embed_dim"]
        self.E = prithvi_params["img_size"] // prithvi_params["patch_size"][-1]
        
    def forward(self, data):
        B, C, T, H, W = data["chip"].shape

        if not self.ablate:
            if self.stride != H:
                Hp, Wp = self.kernel_size
                G = ((H - Hp) // self.stride) + 1
                data["chip"] = torch.functional.F.unfold(data["chip"].view(B, C*T, H, W), kernel_size=self.kernel_size, padding=0, stride=self.stride)
                data["chip"] = data["chip"].view(B, C*T, Hp, Wp, G*G).permute(0, 4, 1, 2, 3).flatten(0, 1).view(B*G*G, C, T, Hp, Wp)       
                if "temporal_coords" in data:
                    data["temporal_coords"] = torch.tile(data["temporal_coords"], (G*G, 1, 1))
                if "location_coords" in data:
                    data["location_coords"] = torch.tile(data["location_coords"], (G*G, 1))

                data = self.prithvi(data)
                data = data.view(B, G, G, self.D, T, self.E, self.E)
            else:
                data = self.prithvi(data)
            
            return data
