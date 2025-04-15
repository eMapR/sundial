import copy
import math
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.models.layers import trunc_normal_, DropPath, Mlp
from torch.nn.init import normal_
from typing import Tuple

from models.multiscale_deformable_attention import MultiScaleDeformableAttention
from models.utils import DoubleConv2d

class PatchEmbed(nn.Module):
    def __init__(
            self,
            input_size: Tuple[int, int, int] = (1, 224, 224),
            patch_size: Tuple[int, int, int] = (1, 16, 16),
            in_chans: int = 6,
            embed_dim: int = 128,
            norm_layer: nn.Module | None = None,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing="ij"
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)

        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]

    return reference_points


def deform_inputs(x, ps, views):
    bs, c, t, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        # [(h, w), (h // 2, w // 2), (h // 4, w // 4), (h // 8, w // 8)],
        [(h//v, w//v) for v in views],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points([(h // ps, w // ps)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // ps, w // ps)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        # [(h, w), (h // 2, w // 2), (h // 4, w // 4), (h // 8, w // 8)],
        [(h//v, w//v) for v in views],
        x.device
    )
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        
        c1s, c2s, c3s = H*W, (H//2)*(W//2), (H//4)*(H//4)
        x1 = x[:, 0:c1s, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x2 = x[:, c1s:c1s+c2s, :].transpose(1, 2).view(B, C, H // 2, W //2).contiguous()
        x3 = x[:, c1s+c2s:c1s+c2s+c3s, :].transpose(1, 2).view(B, C, H // 4, W // 4).contiguous()
        x4 = x[:, c1s+c2s+c3s:, :].transpose(1, 2).view(B, C, H // 8, W // 8).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x4 = self.dwconv(x4).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_frames,
        num_heads=8,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=1.0,
        drop=0.0,
        drop_path=0.0,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer if norm_layer is not None else partial(nn.LayerNorm, eps=1e-6)
        dim *= num_frames

        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(
            embed_dims=dim,
            num_levels=n_levels,
            num_heads=num_heads,
            num_points=n_points,
            value_proj_ratio=deform_ratio,
            batch_first=True,
        )
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        attn = self.attn(
            query=self.query_norm(query),
            reference_points=reference_points,
            value=self.feat_norm(feat),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=None,
        )
        query = query + attn

        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
        return query


class Injector(nn.Module):
    def __init__(
        self,
        dim,  # 1024
        num_frames,
        num_heads=8,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        init_values=0.0,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer if norm_layer is not None else partial(nn.LayerNorm, eps=1e-6)
        dim *= num_frames
    
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(
            embed_dims=dim,
            num_levels=n_levels,
            num_heads=num_heads,
            num_points=n_points,
            value_proj_ratio=deform_ratio,
            batch_first=True,
        )
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        attn = self.attn(
            query=self.query_norm(query),
            reference_points=reference_points,
            value=self.feat_norm(feat),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=None,
        )
        return query + self.gamma * attn


class InteractionBlock(nn.Module):
    def __init__(
        self,
        dim,  # 1024
        num_frames,
        num_heads=8,
        n_points=4,
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=1.0,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer if norm_layer is not None else partial(nn.LayerNorm, eps=1e-6)

        self.injector = Injector(
            dim=dim,
            num_frames=num_frames,
            n_levels=4,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
        )
        self.extractor = Extractor(
            dim=dim,
            num_frames=num_frames,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_frames=num_frames,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None
        self.T = num_frames

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        B, L, D = x.shape

        x = x.reshape(B, self.T, -1, D).permute(0, 2, 1, 3).reshape(B, -1, D*self.T)
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )

        x = x.reshape(B, -1, self.T, D).permute(0, 2, 1, 3).reshape(B, L, D)
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)

        x = x.reshape(B, self.T, -1, D).permute(0, 2, 1, 3).reshape(B, -1, D*self.T)
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W,
                )

        x = x.reshape(B, -1, self.T, D).permute(0, 2, 1, 3).reshape(B, L, D)
        return x, c


class SpatialPriorModule(nn.Module):
    def __init__(self, in_channels: int = 6, inplanes=128, embed_dim=128, num_frames=1):
        super().__init__()

        self.stem = nn.Sequential(
            *[
                nn.Conv3d(in_channels, inplanes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv3d(inplanes, inplanes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv3d(inplanes, inplanes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv3d(
                    inplanes,
                    inplanes,
                    kernel_size=(1,3,3),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False,
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv3d(
                    inplanes,
                    inplanes,
                    kernel_size=(1,3,3),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False,
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv3d(
                    inplanes,
                    inplanes,
                    kernel_size=(1,3,3),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False,
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.fc1 = nn.Conv3d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv3d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv3d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv3d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.T = num_frames

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _, _ = c1.shape
        c1 = c1.reshape(bs, dim, self.T, -1).permute(0, 2, 1, 3).reshape(bs, dim*self.T, -1).transpose(1, 2)
        c2 = c2.reshape(bs, dim, self.T, -1).permute(0, 2, 1, 3).reshape(bs, dim*self.T, -1).transpose(1, 2)
        c3 = c3.reshape(bs, dim, self.T, -1).permute(0, 2, 1, 3).reshape(bs, dim*self.T, -1).transpose(1, 2)
        c4 = c4.reshape(bs, dim, self.T, -1).permute(0, 2, 1, 3).reshape(bs, dim*self.T, -1).transpose(1, 2)

        return c1, c2, c3, c4


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
                 proj_drop=0., window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

    def forward(self, x, H, W):
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size

        qkv = self.qkv(x)  # [B, N, C]
        qkv = qkv.transpose(1, 2).reshape(B, C * 3, H, W)  # [B, C, H, W]
        qkv = F.pad(qkv, [0, W_ - W, 0, H_ - H], mode='constant')

        qkv = F.unfold(qkv, kernel_size=(self.window_size, self.window_size),
                       stride=(self.window_size, self.window_size))
        B, C_kw_kw, L = qkv.shape  # L - the num of windows
        qkv = qkv.reshape(B, C * 3, N_, L).permute(0, 3, 2, 1)  # [B, L, N_, C]
        qkv = qkv.reshape(B, L, N_, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv.unbind(0)

        # q,k,v [B, L, num_head, N_, C/num_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        # attn @ v = [B, L, num_head, N_, C/num_head]
        x = (attn @ v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L)

        x = F.fold(x, output_size=(H_, W_),
                   kernel_size=(self.window_size, self.window_size),
                   stride=(self.window_size, self.window_size))  # [B, C, H_, W_]
        x = x[:, :, :H, :W].reshape(B, C, N).transpose(-1, -2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm=LayerNorm,
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = norm(bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(bottleneck_channels,
                               bottleneck_channels,
                               3,
                               padding=1,
                               bias=False,)
        self.norm2 = norm(bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = norm(out_channels)

        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in [self.conv1, self.norm1, self.act1,
                      self.conv2, self.norm2, self.act2,
                      self.conv3, self.norm3]:
            x = layer(x)

        out = x + out
        return out
    

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., with_cp=False,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 windowed=False, window_size=14, use_residual=False, layer_scale=False):
        super().__init__()
        self.with_cp = with_cp
        self.use_residual = use_residual
        self.norm1 = norm_layer(dim)
        if windowed:
            self.attn = WindowedAttention(dim, num_heads=num_heads,
                                          qkv_bias=qkv_bias, attn_drop=attn_drop,
                                          proj_drop=drop, window_size=window_size)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)
            
        if self.use_residual:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm=LayerNorm,
                act_layer=act_layer,
            )
            
    def forward(self, x, H, W):
        
        def _inner_forward(x):
            if self.layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                
            if self.use_residual:
                B, N, C = x.shape
                x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = self.residual(x)
                x = x.permute(0, 2, 3, 1).reshape(B, N, C)
                
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_size=(1, 224, 224),
                 patch_size=(1, 16, 16),
                 in_chans=6,
                 residual_indices=[],
                 embed_dim=128,
                 depth=12,
                 num_heads=16,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 layer_scale=True,
                 window_attn=False,
                 window_size=14,
                 with_cp=False,
                 norm_layer=None):
        super().__init__()
        self.num_tokens = 1
        self.norm_layer = norm_layer if norm_layer is not None else partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = nn.GELU
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate

        window_attn = [window_attn] * depth if not isinstance(window_attn, list) else window_attn
        window_size = [window_size] * depth if not isinstance(window_size, list) else window_size

        self.patch_embed = PatchEmbed(
            input_size=input_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.dropout = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=self.norm_layer, act_layer=self.act_layer,
                  windowed=window_attn[i], window_size=window_size[i],
                  layer_scale=layer_scale, with_cp=with_cp,
                  use_residual=True if i in residual_indices else False) for i in range(depth)
        ])

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm_layer(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class FuseLayer(nn.Module):
    def __init__(self,
                 embed_dim=128):
        super().__init__()
        self.fuse_layer = DoubleConv2d(embed_dim*2, embed_dim)
        self.norm = nn.SyncBatchNorm(embed_dim)
    
    def forward(self, x, c, scale):
        x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        cat = torch.concat([x, c], dim=1)
        fused = self.fuse_layer(cat)
        return self.norm(fused)


class ViTAdapter(VisionTransformer):
    def __init__(self,
                 input_size=(1, 224, 224),
                 patch_size=(1, 16, 16),
                 in_chans=6,
                 embed_dim=128,
                 num_heads=16,
                 depth=12,
                 conv_inplane=128,
                 n_points=4,
                 deform_num_heads=8,
                 init_values=0.,
                 interaction_indexes=[[0, 1], [1, 2], [2, 3], [3, 4]],
                 with_cffn=True,
                 cffn_ratio=1.0,
                 deform_ratio=1.0,
                 use_extra_extractor=True,
                 fuse=False,
                 *args, **kwargs):
        super().__init__(input_size=input_size,
                         patch_size=patch_size,
                         in_chans=in_chans,
                         embed_dim=embed_dim,
                         num_heads=num_heads,
                         depth=depth,
                         *args, **kwargs)
        self.interaction_indexes = interaction_indexes
        self.embed_dim = embed_dim

        self.level_embed = nn.Parameter(torch.zeros(4, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim,
                                      num_frames=input_size[0])
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_frames=input_size[0],
                             num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor))
            for i in range(len(interaction_indexes))
        ])

        self.fuse_layer1 = FuseLayer(embed_dim)
        self.fuse_layer2 = FuseLayer(embed_dim)
        self.fuse_layer3 = FuseLayer(embed_dim)
        self.fuse_layer4 = FuseLayer(embed_dim)
        
        self.fuse_layer_final = DoubleConv2d(embed_dim*4, embed_dim)        
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _add_level_embed(self, c1, c2, c3, c4):
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        c4 = c4 + self.level_embed[3]
        return c1, c2, c3, c4

    def forward(self, x):
        B, C, T, H, W = x.shape
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_embed.patch_size[-1], [1,2,4,8])

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c1, c2, c3, c4 = self._add_level_embed(c1, c2, c3, c4)
        c1s, c2s, c3s = c1.size(1), c2.size(1), c3.size(1)
        c = torch.cat([c1, c2, c3, c4], dim=1)

        # Patch Embedding forward
        x = self.patch_embed(x)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        bs, _, dim = x.shape
        c1 = c[:, 0:c1s, :].transpose(1, 2).view(bs, dim, H, W).contiguous()
        c2 = c[:, c1s:c1s+c2s, :].transpose(1, 2).view(bs, dim, H//2, W//2).contiguous()
        c3 = c[:, c1s+c2s:c1s+c2s+c3s, :].transpose(1, 2).view(bs, dim, H//4, W//4).contiguous()
        c4 = c[:, c1s+c2s+c3s:, :].transpose(1, 2).view(bs, dim, H//8, W//8).contiguous()

        Hp, Wp = H // self.patch_embed.patch_size[-2], W // self.patch_embed.patch_size[-1]
        x = x.transpose(1, 2).reshape(B, -1, Hp, Wp)
        f1 = self.fuse_layer1(x, c1, 16)
        f2 = self.fuse_layer2(x, c2, 8)
        f3 = self.fuse_layer3(x, c3, 4)
        f4 = self.fuse_layer4(x, c4, 2)

        f4 = F.interpolate(f4, scale_factor=8, mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, scale_factor=4, mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, scale_factor=2, mode="bilinear", align_corners=False)
        
        fused = self.fuse_layer_final(torch.concat([f4, f3, f2, f1], dim=1))
    
        return fused
    

class Model(L.LightningModule):
    def __init__(self, ema_start=128, ema_decay=.97, *t_args, **t_kwargs):
        super().__init__()
        self.model = ViTAdapter(*t_args, **t_kwargs)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_decay = ema_decay
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
    def training_step(self, batch):
        loss = self.criterion(self.model(batch["chip"]), self.ema_model(batch["chip"]))
        return {"loss": loss}

    def validation_step(self, batch):
        output = {"output": self.model(batch["chip"])}
        if self.criterion is not None:
            output["loss"] = self.criterion(output["output"], self.ema_model(batch["chip"]))
        return output 

    def test_step(self, batch):
        output = {"output": self.model(batch["chip"])}
        if self.criterion is not None:
            output["loss"] = self.criterion(output["output"], self.ema_model(batch["chip"]))
        return output 

    def predict_step(self, batch):
        output = {"output": self.model(batch["chip"])}
        return output
    
    def update_ema(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1.0 - self.decay)
    
    def on_batch_end(self):
        if self.global_step > self.ema_start_step:
            self.ema.update(self.model)
    