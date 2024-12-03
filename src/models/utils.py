import torch.nn.functional as F

from torch import nn


class Conv3dBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
            

    def forward(self, x):
        return self.block(x)
    
class ConvTranspose3dBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 16, 16),
                 stride=(1, 16, 16),
                 padding=(0, 0, 0)):
        super().__init__()
        
        self.block = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
                )
    def forward(self, x):
        return self.block(x)


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
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2))

        self.blocks = nn.Sequential(
            *[build_block(embed_dim, embed_dim) for _ in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='bilinear'):
        super(ResizeConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
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
    
class FirstOrderDifference(nn.Module):
    def forward(self, x):
        return x[:, :, 1:, :, :] - x[:, :, :-1, :, :]


class SecondOrderDifference(nn.Module):
    def forward(self, x):
        return x[:, :, 2:, :, :] - 2 * x[:, :, 1:-1, :, :] + x[:, :-2, :, :]