from torch import nn, tensor


class BCEWithLogitsLoss(nn.Module):
    def __init__(self,
                 weight: list[float] = None,
                 size_average: bool = None,
                 reduce: bool = None,
                 reduction: str = 'mean',
                 pos_weight: list[float] = None):
        super().__init__()

        # for config convenience
        if weight is not None:
            weight = tensor(weight)
        if pos_weight is not None:
            pos_weight = tensor(pos_weight)

        self.loss = nn.BCEWithLogitsLoss(weight=weight,
                                         size_average=size_average,
                                         reduce=reduce,
                                         reduction=reduction,
                                         pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.loss(logits, targets)


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 weight: list[float] = None,
                 size_average: bool = None,
                 ignore_index: int = -100,
                 reduce: bool = None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        super().__init__()

        # for config convenience
        if weight is not None:
            weight = tensor(weight)

        self.loss = nn.CrossEntropyLoss(weight=weight,
                                        size_average=size_average,
                                        ignore_index=ignore_index,
                                        reduce=reduce,
                                        reduction=reduction,
                                        label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        return self.loss(logits, targets)
