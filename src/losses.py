from torch import nn
from torchvision.ops import sigmoid_focal_loss


class JacardLoss(nn.Module):
    def __init__(self,
                 smooth: int = 1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + self.smooth)/(union + self.smooth)

        return 1 - IoU


class DiceLoss(nn.Module):
    def __init__(self,
                 smooth: int = 1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth) / \
            (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self,
                 smooth: int = 1):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + self.smooth) / \
            (inputs.sum() + targets.sum() + self.smooth)
        BCE = nn.functional.binary_cross_entropy(
            inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.8,
                 gamma: int = 2,
                 smooth: int = 1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = nn.functional.binary_cross_entropy(
            inputs, targets, reduction='mean')
        BCE_EXP = nn.functional.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE

        return focal_loss


class TVFocalLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: int = 2,
                 reduction: str = 'none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(inputs=inputs,
                                  targets=targets,
                                  alpha=self.alpha,
                                  gamma=self.gamma,
                                  reduction=self.reduction)


class TverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 smooth: int = 1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha *
                                        FP + self.beta*FN + self.smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: int = 1,
                 smooth: int = 1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha *
                                        FP + self.beta*FN + self.smooth)
        FocalTversky = (1 - Tversky)**self.gamma

        return FocalTversky
