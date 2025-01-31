import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import SundialPLBase


class Conv3dBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=(1,3,3),
                stride=1,
                padding=(0,1,1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    

class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv3dBlock(in_channels, mid_channels, kernel_size, stride, padding),
            Conv3dBlock(mid_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Down3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), mid_channels=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2), padding=0),
            DoubleConv3d(in_channels, out_channels, kernel_size, stride, padding, mid_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up3d(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv3d(in_channels + in_channels // 2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
            self.conv = DoubleConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class OutConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 1, 1), stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class UNet3D(SundialPLBase):
    def __init__(self, n_channels, n_classes, bilinear=False, kernel_size=(1,3,3), stride=1, padding=(0,1,1)):
        super().__init__()
        
        self.num_classes = n_classes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3d(n_channels, 64, kernel_size=(1,3,3), stride=1, padding=(0,1,1))
        self.down1 = Down3d(64, 128)
        self.down2 = Down3d(128, 256)
        self.down3 = Down3d(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down3d(512, 1024 // factor)
        self.up1 = Up3d(1024, 512 // factor, bilinear=bilinear)
        self.up2 = Up3d(512, 256 // factor, bilinear=bilinear)
        self.up3 = Up3d(256, 128 // factor, bilinear=bilinear)
        self.up4 = Up3d(128, 64, bilinear=bilinear)
        self.outc = OutConv3d(64, n_classes)

    def forward(self, x):
        x = x["chip"]
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x).squeeze(2)
        return logits