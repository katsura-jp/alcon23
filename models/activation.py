import torch
import torch.nn as nn
import math

class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, input):
        return 0.5 * input * (1 + torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return input * input.sigmoid()