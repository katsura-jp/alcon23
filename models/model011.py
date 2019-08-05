import torch
import torch.nn as nn
from .senet import *

class SEResNeXtGRU2(nn.Module):
    def __init__(self, num_classes, hidden_size=512, bidirectional=False, dropout=0.5, load_weight=None):
        super(SEResNeXtGRU2, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.se_resnet = se_resnext101_32x4d(num_classes=48, pretrained='imagenet', dropout=None)
        if load_weight is not None:
            self.se_resnet.load_state_dict(torch.load(load_weight))

        self.layer0 = self.se_resnet.layer0
        # self.layer0 = nn.Sequential(
        #             nn.Conv2d(3, 64, kernel_size=7, stride=2,padding=3, bias=False),
        #             nn.BatchNorm2d(64),
        #             nn.ReLU(inplace=True),
        #             nn.MaxPool2d(3, stride=2,ceil_mode=True)
        #         )
        self.layer1 = self.se_resnet.layer1
        self.layer2 = self.se_resnet.layer2
        self.layer3 = self.se_resnet.layer3
        self.layer4 = self.se_resnet.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

        if bidirectional:
            self.gru1 = nn.GRU(self.se_resnet.last_linear.in_features, hidden_size=hidden_size//2, batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(hidden_size, hidden_size=hidden_size//2, batch_first=True,bidirectional=True)
        else:
            self.gru1 = nn.GRU(self.se_resnet.last_linear.in_features, hidden_size=hidden_size, batch_first=True, bidirectional=False)
            self.gru2 = nn.GRU(hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)

        self.fc = nn.Linear(hidden_size, num_classes)

    def encode(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x

    def decode(self, x):
        batch, seq, _ = x.size()
        if self.bidirectional:
            hidden = torch.zeros(2, batch, self.hidden_size//2, dtype=x.dtype, device=x.device)
        else:
            hidden = torch.zeros(1, batch, self.hidden_size, dtype=x.dtype, device=x.device)

        x, _ = self.gru1(x, hidden)
        x, _ = self.gru2(x, hidden)

        x = self.fc(x.contiguous().view(-1, x.size(2)))
        return x

    def forward(self, x):
        b, s, c, h, w = x.size()
        x = self.encode(x.view(-1, c, h, w))
        x = self.decode(x.view(b, s, -1))
        return x



def test():
    inputs = torch.FloatTensor(8, 3, 3, 224, 224)
    model = SEResNeXtGRU2(48, 512, bidirectional=True)
    logits = model(inputs)
    print(logits.size())

if __name__ == '__main__':
    test()
