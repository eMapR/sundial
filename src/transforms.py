import datetime
import torch
import torchvision

from torch import nn


class GeoNormalization(nn.Module):
    def __init__(self,
                 means: list[float],
                 stds: list[float]):
        super().__init__()
        self.means = torch.tensor(means, dtype=torch.float).view(-1, 1, 1, 1)
        self.stds = torch.tensor(stds, dtype=torch.float).view(-1, 1, 1, 1)

    def forward(self, x):
        x = (x - self.means) / self.stds
        return x


class ReverseTime(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.flip(self.dim)


class Flatten(nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)


class FirstOrderDifference(nn.Module):
    def forward(self, x):
        return x[:, 1:, :, :] - x[:, :-1, :, :]


class SecondOrderDifference(nn.Module):
    def forward(self, x):
        return x[:, 2:, :, :] - 2 * x[:, 1:-1, :, :] + x[:, :-2, :, :]


class BinaryStep(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, 1.0, 0.0)
    

class BeforeAfter(nn.Module):
    def forward(self, x):
        return torch.stack([x[:,-3,:,:], x[:,-1,:,:]], dim=1)

class SingleClass(nn.Module):
    def __init__(self, class_index: int):
        super().__init__()
        self.class_index = class_index
    
    def forward(self, x):
        return x[self.class_index,...].unsqueeze(0)
    

class RandomAffineAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        rot0 = torchvision.transforms.v2.RandomRotation((0, 0))
        rot90 = torchvision.transforms.v2.RandomRotation((90, 90))
        rot180 = torchvision.transforms.v2.RandomRotation((180, 180))
        rot270 = torchvision.transforms.v2.RandomRotation((270, 270))
        
        self.rotation = torchvision.transforms.v2.RandomChoice([rot0, rot90, rot180, rot270])
        self.hflip = torchvision.transforms.v2.RandomHorizontalFlip()
        self.vflip = torchvision.transforms.v2.RandomVerticalFlip()
    
    def forward(self, x):
        x = self.rotation(x)
        x = self.hflip(x)
        x = self.vflip(x)
        return x

   
class RandomCropAndAffine(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.affine = RandomAffineAugmentation()
        self.crop = torchvision.transforms.v2.RandomCrop(size=size)
        
    def forward(self, x):
        x = self.crop(x)
        x = self.affine(x)
        return x


class GeoColorJitter(nn.Module):
    def __init__(self,
                 brightness: float | tuple[float, float] = 0,
                 contrast: float | tuple[float, float] = 0,
                 saturation: float | tuple[float, float] = 0,
                 hue: float | tuple[float, float] = 0,
                 uniform: bool = True):
        super().__init__()
        self.color_jitter = torchvision.transforms.v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.uniform = uniform

    def forward(self, x):
        num_bands = x.shape[0]
        num_frames = x.shape[1]
        seed = datetime.datetime.now().timestamp()

        stack = []
        for t in range(num_frames):
            cat = []
            for b in range(num_bands):
                if self.uniform:
                    torch.manual_seed(seed=seed)
                cat.append(self.color_jitter(x[b, t].unsqueeze(0)))
            stack.append(torch.cat(cat))
        return torch.stack(stack, dim=1)


class CatNDVI(nn.Module):
    def __init__(self,
                 nir_band_idx: int = 3,
                 red_band_idx: int = 2):
        super().__init__()
        self.nir_band_idx = nir_band_idx
        self.red_band_idx = red_band_idx

    def forward(self, x):
        nir = x[:, :, self.nir_band_idx, :, :]
        red = x[:, :, self.red_band_idx, :, :]
        ndvi = (nir - red) / (nir + red)
        return torch.cat([x, ndvi.unsqueeze(2)], dim=2)


class CatNDWI(nn.Module):
    def __init__(self,
                 nir_band_idx: int = 3,
                 swir_band_idx: int = 5):
        super().__init__()
        self.nir_band_idx = nir_band_idx
        self.swir_band_idx = swir_band_idx

    def forward(self, x):
        nir = x[:, :, self.nir_band_idx, :, :]
        swir = x[:, :, self.swir_band_idx, :, :]
        ndwi = (nir - swir) / (nir + swir)
        return torch.cat([x, ndwi.unsqueeze(2)], dim=2)


class CatNBR(nn.Module):
    def __init__(self,
                 nir_band_idx: int = 3,
                 swir_band_idx: int = 5):
        super().__init__()
        self.nir_band_idx = nir_band_idx
        self.swir_band_idx = swir_band_idx

    def forward(self, x):
        nir = x[:, :, self.nir_band_idx, :, :]
        swir = x[:, :, self.swir_band_idx, :, :]
        nbr = (nir - swir) / (nir + swir)
        return torch.cat([x, nbr.unsqueeze(2)], dim=2)
