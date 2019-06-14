import torch
import torch.nn as nn
from .senet import *


class SE_ResNextLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=512, bidirectional=False):
        super(SE_ResNextLSTM, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.resnext = se_resnext101_32x4d()
        self.layer0 = self.resnext.layer0
        self.layer1 = self.resnext.layer1
        self.layer2 = self.resnext.layer2
        self.layer3 = self.resnext.layer3
        self.layer4 = self.resnext.layer4
        self.avg_pool = self.resnext.avg_pool
        self.dropout = self.resnext.dropout

        if bidirectional:
            self.lstm = nn.LSTM(self.resnext.last_linear.in_features, hidden_size=hidden_size//2, batch_first=True,bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.resnext.last_linear.in_features, hidden_size=hidden_size, batch_first=True, bidirectional=False)

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
            hidden = (torch.zeros(2, batch, self.hidden_size//2, dtype=x.dtype, device=x.device),
                        torch.zeros(2, batch, self.hidden_size//2, dtype=x.dtype, device=x.device))
        else:
            hidden = (torch.zeros(1, batch, self.hidden_size, dtype=x.dtype, device=x.device),
                        torch.zeros(1, batch, self.hidden_size, dtype=x.dtype, device=x.device))

        x, _ = self.lstm(x, hidden)
        x = self.fc(x.contiguous().view(-1, x.size(2))).view(batch, seq, -1)
        return x

    def forward(self, x):
        b, s, c, h, w = x.size()
        x = self.encode(x.view(-1, c, h, w))
        x = self.decode(x.view(b, s, -1))
        return x



def test():
    inputs = torch.FloatTensor(8, 3, 3, 224, 224)
    model = SE_ResNextLSTM(10, 512, bidirectional=True)
    logits = model(inputs)
    print(logits.size())

if __name__ == '__main__':
    test()