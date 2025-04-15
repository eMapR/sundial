import torch

from torch import nn, einsum
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.ops import sigmoid_focal_loss
from typing import Any, Literal, Sequence

from utils import distance_transform


class Base(nn.Module):
    def __init__(self,
                 logits: bool = True,
                 multiclass: bool = False,
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        self.logits = logits
        self.multiclass = multiclass
        self.reducer = Reducer(reduction)
    
    def forward_activation(self, inputs):
        if self.logits:
            if self.multiclass:
                inputs = nn.functional.softmax(inputs, dim=1)
            else:
                inputs = nn.functional.sigmoid(inputs)
        return inputs

    def forward(self, inputs, targets):
        inputs = self.forward_activation(inputs)
        loss = self.forward_loss(inputs, targets)
        loss = self.reducer(loss)
        return loss


class Reducer(nn.Module):
    def __init__(self,
                 reduction: str):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs):
        match self.reduction:
            case "none":
                return torch.identity(inputs)
            case "mean":
                return torch.mean(inputs)
            case "sum":
                return torch.sum(inputs)


class RMSELoss(nn.Module):
    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(size_average, reduce, reduction)
        
    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))


class JaccardLoss(Base):
    def __init__(self,
                 epsilon: int = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward_loss(self, inputs, targets):
        intersection = einsum("bcwh,bcwh->bc", inputs, targets)
        union = einsum("bkwh->bk", inputs) + \
            einsum("bkwh->bk", targets) - intersection

        jaccard = (intersection + self.epsilon)/(union + self.epsilon)

        return 1 - einsum("bk->b", jaccard)


class DiceLoss(Base):
    def __init__(self,
                 epsilon: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
            
    def forward_loss(self, inputs, targets):
        intersection = einsum("bcwh,bcwh->bc", inputs, targets)
        sum_probs = einsum("bcwh->bc", inputs) + einsum("bcwh->bc", targets)
        loss = (2. * intersection + self.epsilon) / (sum_probs + self.epsilon)  

        return 1 - loss


class GeneralizedDiceLoss(Base):
    def __init__(self,
                 epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward_loss(self, inputs, targets):
        weight = 1 / ((einsum("bkwh->bk", targets) + self.epsilon) ** 2)
        intersection = weight * einsum("bcwh,bcwh->bc", inputs, targets)
        sum_probs = weight * einsum("bkwh->bk", inputs) + einsum("bkwh->bk", targets)

        generalized_dice = (2. * einsum("bk->b", intersection) + self.epsilon) / (einsum("bk->b", sum_probs) + self.epsilon)

        return 1 - generalized_dice


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


class TverskyLoss(Base):
    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 epsilon: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward_loss(self, inputs, targets):
        TP = einsum("bcwh,bcwh->b", inputs, targets)
        FP = einsum("bcwh,bcwh->b", inputs, 1-targets)
        FN = einsum("bcwh,bcwh->b", targets, 1-inputs)

        tversky = (TP + self.epsilon) / (TP + self.alpha *
                                         FP + self.beta*FN + self.epsilon)

        return 1 - tversky


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
                 label_smoothing: float = 0,
                 device: torch.device | None = None):
        if weight is not None:
            weight = torch.tensor(weight, device=device, dtype=torch.float)
        super().__init__(weight=weight,
                         size_average=size_average,
                         ignore_index=ignore_index,
                         reduce=reduce,
                         reduction=reduction,
                         label_smoothing=label_smoothing)


class SSIMLoss(Base):
    def __init__(self,
                 gaussian_kernel: bool = True,
                 sigma: float | Sequence[float] = 1.5,
                 kernel_size: int | Sequence[int] = 11,
                 reduction: Literal['elementwise_mean',
                                    'sum', 'none'] | None = "elementwise_mean",
                 data_range: float | tuple[float, float] | None = None,
                 k1: float = 0.01,
                 k2: float = 0.03):
        super().__init__(reduction=None, )
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

    def forward_loss(self, inputs, targets):
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


class DiceBoundaryLoss(Base):
    def __init__(self,
                 alpha: float = 0.01,
                 epsilon: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.dice = DiceLoss(epsilon=epsilon, logits=False, reduction=None)

    def forward_loss(self, inputs, targets):
        distances = torch.stack([distance_transform(targets[b])
                                for b in range(targets.shape[0])])
                
        boundary_loss = einsum('bcwh,bcwh->bcwh', inputs, distances).mean()
        loss = self.dice(inputs, targets) + self.alpha * boundary_loss
        return loss


class GeneralzedDiceBoundaryLoss(Base):
    def __init__(self,
                 alpha: float = 0.01,
                 epsilon: float = 1e-6,
                 ):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.gdl = GeneralizedDiceLoss(epsilon=epsilon, logits=False, reduction=None)

    def forward_loss(self, inputs, targets):
        distances = torch.stack([distance_transform(targets[b])
                                for b in range(targets.shape[0])])

        boundary_loss = einsum('bcwh,bcwh->bcwh', inputs, distances).mean()
        loss = self.gdl(inputs, targets) + self.alpha * boundary_loss
        return loss
    
    
class DiceCrossEntropyLoss(Base):
    def __init__(self,
                 epsilon: float = 1e-6,
                 weight: list[float] | None = None,
                 size_average: Any | None = None,
                 ignore_index: int = -100,
                 
                 reduce: Any | None = None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0,
                 device: torch.device | None = None,
                 logits: bool = True,
                 multiclass: bool = False):
        super().__init__()
        self.epsilon = epsilon
        self.dice = DiceLoss(epsilon=epsilon, logits=logits, multiclass=multiclass, reduction=reduction)
        self.ce = CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
            device=device
        )
    
    def forward(self, inputs, targets):
        loss = self.dice(inputs, targets) + self.ce(inputs, targets)
        return loss


class InfoNCE(nn.Module):
    """
    https://arxiv.org/abs/1807.03748v2
    https://arxiv.org/abs/2010.05113

    """

    def __init__(self, temperature=0.1, reduction='mean', mode='intra'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.mode = mode

    def forward(self, query, positive_key):
        return self.info_nce(query, positive_key,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        mode=self.mode)

    def info_nce(self, query, positive_key, temperature=0.1, reduction='mean', mode='intra'):
        N, D, H, W = query.shape
        query = query.permute(0, 2, 3, 1)
        positive_key = positive_key.permute(0, 2, 3, 1)
        
        if query.shape != positive_key.shape:
            raise ValueError(f"<query> and <positive_key> must be same shape. Got {query.shape} vs {positive_key.shape}")

        # inter requires O((NHW)²) and is extremely memory intensive. intra is a more memory efficent albeit slower method
        if mode == 'intra':
            loss = 0
            for i in range(N):
                q = query[i].reshape(-1, D) # (H*W, D)
                k = positive_key[i].reshape(-1, D) # (H*W, D)

                q, k = self.normalize(q, k)
                logits = q @ k.T  / temperature  # (H*W, H*W)
                labels = torch.arange(logits.shape[0], device=query.device)
                loss += F.cross_entropy(logits, labels)
            return loss / N
        elif mode == 'inter':
            query = query.reshape(-1, D) # (N*H*W, D)
            positive_key = positive_key.reshape(-1, D) # (N*H*W, D)

            query, positive_key = self.normalize(query, positive_key)
            logits = query @ positive_key.T / temperature
            labels = torch.arange(query.shape[0], device=query.device)
            
            return F.cross_entropy(logits, labels, reduction=reduction)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]