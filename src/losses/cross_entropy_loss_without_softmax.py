import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn._reduction as _Reduction
from torch._jit_internal import weak_script

@weak_script
def cross_entropy_without_softmax(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return F.nll_loss(torch.log(input), target, weight, None, ignore_index, None, reduction)


class CrossEntropyLossWithoutSoftmax(nn.CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLossWithoutSoftmax, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return cross_entropy_without_softmax(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

