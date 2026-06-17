import gc
import lightning as L
import os
import sys
import torch
import torch.nn.functional as F

from huggingface_hub import snapshot_download
from torch import nn
from transformers import ViTConfig, ViTModel

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
        
        PRITHVI_DIR = snapshot_download(repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M")
        sys.path.append(PRITHVI_DIR)
        from prithvi_mae import PrithviMAE
        
        self.model = PrithviMAE(**self.prithvi_params)
        if self.prithvi_ckpt_path is not None:
            pt_path = os.path.join(PRITHVI_DIR, self.prithvi_ckpt_path)
            checkpoint = torch.load(pt_path, weights_only=False)
            _ = self.model.load_state_dict(checkpoint, strict=True)
        del self.model.decoder
        
        self.reshaper = PrithviReshape(prithvi_params["patch_size"], prithvi_params["img_size"]) if reshape else nn.Identity()
        if self.freeze_encoder:
            for blk in self.model.encoder.blocks:
                for param in blk.parameters():
                    param.requires_grad = False
            

    def forward(self, data):
        if isinstance(data, dict):
            chip = data.get("inpt")
            temporal = data.get("temporal_coords")
            location = data.get("location_coords")
        else:
            chip = data
            temporal = None
            location = None

        latent = self.model.forward_features(chip,
                                             temporal,
                                             location)

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
                 bayesian: bool = False,
                 embed: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ablate = ablate
        self.bayesian = bayesian
        self.embed = embed
        
        if self.ablate:
            self.prithvi =  ViTModel(ViTConfig(**prithvi_params))
            self.patch_size = prithvi_params["patch_size"]
            dim = prithvi_params["hidden_size"]*prithvi_params["num_frames"]
        else:
            self.prithvi = PrithviBackbone(
                prithvi_params=prithvi_params,
                freeze_encoder=freeze_encoder,
                prithvi_ckpt_path=prithvi_ckpt_path,
                reshape=True)
            dim = prithvi_params["embed_dim"]*prithvi_params["num_frames"]

        
        if not self.embed:
            from torchvision.models.segmentation.fcn import FCNHead
            from models.utils import Upscaler
            self.head = nn.Sequential(
                Upscaler(dim, 4),
                FCNHead(dim // 2**4, self.num_classes)
            )
            
        if bayesian:
            from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
            const_bnn_prior_parameters = {
                    "prior_mu": 0.0,
                    "prior_sigma": 1.0,
                    "posterior_mu_init": 0.0,
                    "posterior_rho_init": -3.0,
                    "type": "Reparameterization",
                    "moped_enable": True,
                    "moped_delta": 0.5,
            }
            dnn_to_bnn(self.head, const_bnn_prior_parameters)
            
        
    def forward(self, data):
        if self.ablate:
            B, C, T, H, W = data["inpt"].shape
            Hp, Wp = H // self.patch_size, W // self.patch_size
            latent = []
            for i in range(T):
                time_step = data["inpt"][:,:,i,...]
                l = self.prithvi(time_step).last_hidden_state[:,1:,...].transpose(1,2).reshape(B, -1, Hp, Wp)
                latent.append(l)
            latent = torch.concat(latent, dim=1)
        else:
            latent = self.prithvi(data)
        if self.embed:
            return latent
        else:
            predictions = self.head(latent)
            return predictions


class PrithviFCNMosaic(SundialPLBase):
    def __init__(self,
                 num_classes: int,
                 upscale_depth: int,
                 prithvi_params: dict,
                 freeze_encoder: bool = True,
                 prithvi_ckpt_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.upscale_depth = upscale_depth
        self.kernel_size = prithvi_params["img_size"]
        
        self.prithvi = PrithviBackbone(
            prithvi_params=prithvi_params,
            freeze_encoder=freeze_encoder,
            prithvi_ckpt_path=prithvi_ckpt_path,
            reshape=True)
        
        from torchvision.models.segmentation.fcn import FCNHead
        from models.utils import Upscaler
        dim = prithvi_params["embed_dim"]*prithvi_params["num_frames"]
        self.head = nn.Sequential(
            Upscaler(dim, self.upscale_depth),
            FCNHead(dim // 2**self.upscale_depth, self.num_classes)
        )
            
    def forward(self, data):
        B, C, T, H, W = data["inpt"].shape
        assert H % self.kernel_size == 0
        
        N = (H // self.kernel_size)*(W // self.kernel_size)
        H_ = W_ = self.kernel_size
        B_ = B * N
        
        imgs = torch.functional.F.unfold(data["inpt"].view(B,C*T,H,W), kernel_size=self.kernel_size, stride=self.kernel_size, padding=0)
        imgs = imgs.permute(0, 2, 1).reshape(B_, C, T, H_, W_)
        
        latent = self.prithvi(imgs)
        predictions = self.head(latent)
        predictions = predictions.reshape(B, N, self.num_classes*H_*W_).permute(0, 2, 1)
        predictions = torch.functional.F.fold(predictions, output_size=(H, W), kernel_size=self.kernel_size, stride=self.kernel_size, padding=0)
        
        return predictions


class PrithviFCNDelta(SundialPLBase):
    def __init__(self,
                 num_classes: int,
                 num_frames: int,
                 upscale_depth: int,
                 method: str,
                 separate: bool,
                 prithvi_params: dict,
                 freeze_encoder: bool = True,
                 prithvi_ckpt_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.upscale_depth = upscale_depth
        self.separate = separate
        self.method = method
        self.kernel_size = prithvi_params["img_size"]
        
        self.prithvi = PrithviBackbone(
            prithvi_params=prithvi_params,
            freeze_encoder=freeze_encoder,
            prithvi_ckpt_path=prithvi_ckpt_path,
            reshape=True)
        
        from torchvision.models.segmentation.fcn import FCNHead
        from models.utils import Upscaler

        dim = prithvi_params["embed_dim"]*num_frames
        
        self.head = nn.Sequential(
            Upscaler(dim, self.upscale_depth),
            FCNHead(dim // 2**(self.upscale_depth+2), self.num_classes)
        )
            
    def forward(self, data):
        B, C, T, H, W = data["inpt"].shape
        latent = []
        if self.separate:
            for t in range(T):
                img = data["inpt"][:,:,t].unsqueeze(2)
                latent.append(self.prithvi(img))
            latent = torch.concat(latent, dim=1)
        else:
            latent = self.prithvi(data)
        B, D, Hp, Wp = latent.shape
        D_ = D // T
        match self.method:
            case 'delta':
                latent = latent[:,:-D_,:,:] - latent[:,D_:,:,:]
            case 'cat_delta':
                latent = torch.cat([latent[:,:D_,:,:], latent[:,:-D_,:,:] - latent[:,D_:,:,:]], dim=1)
            case 'stack':
                pass
        predictions = self.head(latent)
        
        return predictions


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
        super().__init__(**kwargs)
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

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        inpt = inpt["inpt"]
        B, C, T, H, W = inpt.shape

        # deform_inputsN = [reference_points, spatial_shapes, level_start_index]
        deform_inputs1, deform_inputs2 = deform_inputs(x=inpt, ps=self.Ps)

        # Spatial Prior Module (SPM) forward
        c1, c2, c3, c4 = self.spm(inpt)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x = self.backbone.model.encoder.patch_embed(inpt)
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
                 ablate: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
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
        B, C, T, H, W = data["inpt"].shape

        if not self.ablate:
            if self.stride != H:
                Hp, Wp = self.kernel_size
                G = ((H - Hp) // self.stride) + 1
                data["inpt"] = torch.functional.F.unfold(data["inpt"].view(B, C*T, H, W), kernel_size=self.kernel_size, padding=0, stride=self.stride)
                data["inpt"] = data["inpt"].view(B, C*T, Hp, Wp, G*G).permute(0, 4, 1, 2, 3).flatten(0, 1).view(B*G*G, C, T, Hp, Wp)       
                if "temporal_coords" in data:
                    data["temporal_coords"] = torch.tile(data["temporal_coords"], (G*G, 1, 1))
                if "location_coords" in data:
                    data["location_coords"] = torch.tile(data["location_coords"], (G*G, 1))

                data = self.prithvi(data)
                data = data.view(B, G, G, self.D, T, self.E, self.E)
            else:
                data = self.prithvi(data)
            
            return data
