import torch

from torch import nn, einsum
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.ops import sigmoid_focal_loss
from typing import Any, Literal, Sequence

from utils import distance_transform


class Reducer(nn.Module):
    def __init__(self,
                 reduction: str,
                 dim: int | tuple[int] = 0,
                 keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        match reduction:
            case "none":
                self.reduction = torch.identity
            case "mean":
                self.reduction = torch.mean
            case "sum":
                self.reduction = torch.sum

    def forward(self, inputs):
        return self.reduction(inputs, self.dim, self.keepdim)


class RMSELoss(nn.Module):
    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(size_average, reduce, reduction)
        
    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))


class JaccardLoss(nn.Module):
    def __init__(self,
                 epsilon: int = 1e-6,
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reducer = Reducer(reduction)

    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)

        intersection = einsum("bcwh,bcwh->bc", inputs, targets)
        union = einsum("bkwh->bk", inputs) + \
            einsum("bkwh->bk", targets) - intersection

        jaccard = (intersection + self.epsilon)/(union + self.epsilon)

        return self.reducer(1 - einsum("bk->b", jaccard))


class DiceLoss(nn.Module):
    def __init__(self,
                 epsilon: float = 1e-6,
                 logits: bool = True,
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.logits = logits
        self.reducer = Reducer(reduction)

    def forward(self, inputs, targets):
        if self.logits:
            inputs = nn.functional.sigmoid(inputs)

        intersection = einsum("bcwh,bcwh->bc", inputs, targets)
        sum_probs = einsum("bkwh->bk", inputs) + einsum("bkwh->bk", targets)
        dice = (2. * intersection + self.epsilon) / (sum_probs + self.epsilon)

        return self.reducer(1 - einsum("bk->b", dice))


class GeneralizedDiceLoss(nn.Module):
    def __init__(self,
                 epsilon: float = 1e-6,
                 logits: bool = True,
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.logits = logits
        self.reducer = Reducer(reduction)

    def forward(self, inputs, targets):
        if self.logits:
            inputs = nn.functional.sigmoid(inputs)

        weight = 1 / ((einsum("bkwh->bk", targets) + self.epsilon) ** 2)
        intersection = weight * einsum("bcwh,bcwh->bc", inputs, targets)
        sum_probs = weight * einsum("bkwh->bk", inputs) + \
            einsum("bkwh->bk", targets)

        generalized_dice = (2. * einsum("bk->b", intersection) + self.epsilon) / \
            (einsum("bk->b", sum_probs) + self.epsilon)

        return self.reducer(1 - generalized_dice)


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: int = 2,
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = sigmoid_focal_loss(inputs=inputs,
                                  targets=targets,
                                  alpha=self.alpha,
                                  gamma=self.gamma,
                                  reduction=self.reduction)
        return loss


class TverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 epsilon: float = 1e-6,
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.reducer = Reducer(reduction)

    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)

        TP = einsum("bcwh,bcwh->b", inputs, targets)
        FP = einsum("bcwh,bcwh->b", inputs, 1-targets)
        FN = einsum("bcwh,bcwh->b", targets, 1-inputs)

        tversky = (TP + self.epsilon) / (TP + self.alpha *
                                         FP + self.beta*FN + self.epsilon)

        return self.reducer(1 - tversky)


class FocalTverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: int = 1,
                 epsilon: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.tversky = TverskyLoss(
            alpha=alpha, beta=beta, epsilon=epsilon, reduction='none'
        )

    def forward(self, inputs, targets):
        tversky = self.tversky(inputs, targets)
        return tversky**self.gamma


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self,
                 weight: list[float] | None = None,
                 size_average: Any | None = None,
                 reduce: Any | None = None,
                 reduction: str = 'mean',
                 pos_weight: list[float] | None = None,
                 device: torch.device | None = None):
        if weight is not None:
            weight = torch.tensor(weight, device=device, dtype=torch.float)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, device=device, dtype=torch.float)
        super().__init__(weight=weight,
                         size_average=size_average,
                         reduce=reduce,
                         reduction=reduction,
                         pos_weight=pos_weight)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self,
                 weight: list[float] | None = None,
                 size_average: Any | None = None,
                 ignore_index: int = -100,
                 reduce: Any | None = None,
                 reduction: str = 'mean',
                 label_epsiloning: float = 0,
                 device: torch.device | None = None):
        if weight is not None:
            weight = torch.tensor(weight, device=device, dtype=torch.float)
        super().__init__(weight=weight,
                         size_average=size_average,
                         ignore_index=ignore_index,
                         reduce=reduce,
                         reduction=reduction,
                         label_epsiloning=label_epsiloning)


class SSIMLoss(nn.Module):
    def __init__(self,
                 gaussian_kernel: bool = True,
                 sigma: float | Sequence[float] = 1.5,
                 kernel_size: int | Sequence[int] = 11,
                 reduction: Literal['elementwise_mean',
                                    'sum', 'none'] | None = "elementwise_mean",
                 data_range: float | tuple[float, float] | None = None,
                 k1: float = 0.01,
                 k2: float = 0.03):
        super().__init__()
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)
        loss = structural_similarity_index_measure(preds=inputs,
                                                   target=targets,
                                                   gaussian_kernel=self.gaussian_kernel,
                                                   sigma=self.sigma,
                                                   kernel_size=self.kernel_size,
                                                   reduction=self.reduction,
                                                   data_range=self.data_range,
                                                   k1=self.k1,
                                                   k2=self.k2)
        return 1 - loss


class DiceBoundaryLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.01,
                 epsilon: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.dice = DiceLoss(epsilon=epsilon, logits=False)

    def forward(self, inputs, targets):
        distances = torch.stack([distance_transform(targets[b])
                                for b in range(targets.shape[0])])
        inputs = nn.functional.sigmoid(inputs)

        boundary_loss = einsum('bcwh,bcwh->bcwh', inputs, distances).mean()
        loss = self.dice(inputs, targets) + self.alpha * boundary_loss
        return loss


class GeneralzedDiceBoundaryLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.01,
                 epsilon: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.gdl = GeneralizedDiceLoss(epsilon=epsilon, logits=False)

    def forward(self, inputs, targets):
        distances = torch.stack([distance_transform(targets[b])
                                for b in range(targets.shape[0])])
        inputs = nn.functional.sigmoid(inputs)

        boundary_loss = einsum('bcwh,bcwh->bcwh', inputs, distances).mean()
        loss = self.gdl(inputs, targets) + self.alpha * boundary_loss
        return loss
