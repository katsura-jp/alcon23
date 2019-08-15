import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class TemperatureSoftmax(nn.Module):
    __constants__ = ['dim']
    def __init__(self, dim=None):
        super(TemperatureSoftmax, self).__init__()
        self.dim = dim
        self.t = Parameter(torch.FloatTensor([1.]))

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return F.softmax(input / self.t, self.dim, _stacklevel=5)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)