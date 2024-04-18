import datetime
import torch
import torchvision

from torch import nn


class GeoNormalization(nn.Module):
    def __init__(self,
                 means: list[float],
                 stds: list[float]):
        super().__init__()
        self.means = torch.tensor(
            means, dtype=torch.float).view(-1, 1, 1, 1)
        self.stds = torch.tensor(
            stds, dtype=torch.float).view(-1, 1, 1, 1)

    def forward(self, x):
        return (x - self.means) / self.stds


class TimeStepDifference(nn.Module):
    def forward(self, x):
        return x[:, 1:, :, :] - x[:, :-1, :, :]


class GeoColorJitter(nn.Module):
    def __init__(self,
                 brightness: float | tuple[float, float] = 0,
                 contrast: float | tuple[float, float] = 0,
                 saturation: float | tuple[float, float] = 0,
                 hue: float | tuple[float, float] = 0,
                 uniform: bool = True):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
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
