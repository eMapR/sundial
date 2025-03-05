import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import SundialPLBase


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


class Down2d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2d(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2d(in_channels + in_channels // 2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class UNet2D(SundialPLBase):
    def __init__(self, n_channels, n_classes, bilinear=False, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.num_classes = n_classes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv2d(n_channels, 64)
        self.down1 = Down2d(64, 128)
        self.down2 = Down2d(128, 256)
        self.down3 = Down2d(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down2d(512, 1024 // factor)
        self.up1 = Up2d(1024, 512 // factor, bilinear=bilinear)
        self.up2 = Up2d(512, 256 // factor, bilinear=bilinear)
        self.up3 = Up2d(256, 128 // factor, bilinear=bilinear)
        self.up4 = Up2d(128, 64, bilinear=bilinear)
        self.outc = OutConv2d(64, n_classes)

    def forward(self, x):
        x = x["chip"]
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
