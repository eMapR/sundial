import torch
import torch.nn.functional as F

from functools import partial
from torch import nn

from models.base import SundialPLBase


class Conv3dBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, mid_channels=None, embed=False):
        super().__init__()
        self.embed = embed
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv3dBlock(in_channels, mid_channels, kernel_size, stride, padding),
            Conv3dBlock(mid_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        x = self.double_conv(x)
        if self.embed:
            B, C, T, H, W = x.shape
            x = x.reshape(B, C*T, H, W)
            return x
        else:
            return x


class DoubleConv3dMod(DoubleConv3d, SundialPLBase):
    def forward(self, x):
        return super().forward(x["chip"])


class Upscaler(nn.Module):
    def __init__(self, embed_dim: int, depth: int, dropout: bool = True):
        super().__init__()

        def build_block(in_ch, out_ch): return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(0.1) if dropout else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.upscale_blocks = nn.Sequential(
            *[build_block(int(embed_dim // 2**i), int(embed_dim // 2**(i+1))) for i in range(depth)]
        )

    def forward(self, x):
        return self.upscale_blocks(x)


class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super(ResizeConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.block(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_channels=1024, out_channels=64):
        super().__init__()

        self.upsample1 = ResizeConv2d(in_channels, 512, kernel_size=3, scale_factor=2)
        self.upsample2 = ResizeConv2d(512, 256, kernel_size=3, scale_factor=2) 
        self.upsample3 = ResizeConv2d(256, 128, kernel_size=3, scale_factor=2)
        self.upsample4 = ResizeConv2d(128, out_channels, kernel_size=3, scale_factor=2)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        return x


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)

        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]

    return reference_points


def deform_inputs(x, ps):
    bs, c, t, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
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
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device
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
        n = N // 21
        x1 = x[:, 0 : 16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2)
        x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).view(B, C, H, W)
        x3 = x[:, 20 * n :, :].transpose(1, 2).view(B, C, H // 2, W // 2)
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_frames,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer='layernorm',
    ):
        super().__init__()
        if norm_layer == 'layernorm': # partials don't play well with json argparse
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dim *= num_frames

        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        from models.multiscale_deformable_attention import MultiScaleDeformableAttention
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
        norm_layer='layernorm',
        init_values=0.0,
    ):
        super().__init__()
        if norm_layer == 'layernorm': # partials don't play well with jsonargparse
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dim *= num_frames
    
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        from models.multiscale_deformable_attention import MultiScaleDeformableAttention
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
        norm_layer='layernorm',
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
    ):
        super().__init__()
        if norm_layer == 'layernorm': # partials don't play well with jsonargparse
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.injector = Injector(
            dim=dim,
            num_frames=num_frames,
            n_levels=3,
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
        self.D = dim*num_frames
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
            x = blk(x)

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
    def __init__(self, in_channels: int = 3, inplanes=64, embed_dim=384, num_frames=2):
        super().__init__()

        self.stem = nn.Sequential(
            *[
                nn.Conv3d(in_channels, inplanes, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv3d(inplanes, inplanes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv3d(inplanes, inplanes, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv3d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=(1,3,3),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False,
                ),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv3d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=(1,3,3),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv3d(
                    4 * inplanes,
                    4 * inplanes,
                    kernel_size=(1,3,3),
                    stride=(1,2,2),
                    padding=(0,1,1),
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.fc1 = nn.Conv3d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv3d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv3d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv3d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
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
        c1 = c1.reshape(bs, dim, self.T, -1).permute(0, 2, 1, 3).reshape(bs, dim*self.T, -1).transpose(1, 2)  # 4s
        c2 = c2.reshape(bs, dim, self.T, -1).permute(0, 2, 1, 3).reshape(bs, dim*self.T, -1).transpose(1, 2)  # 8s
        c3 = c3.reshape(bs, dim, self.T, -1).permute(0, 2, 1, 3).reshape(bs, dim*self.T, -1).transpose(1, 2)  # 16s
        c4 = c4.reshape(bs, dim, self.T, -1).permute(0, 2, 1, 3).reshape(bs, dim*self.T, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4
