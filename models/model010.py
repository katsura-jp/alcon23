import torch
import torch.nn as nn
from oct_resnet import *

class OctResNetGRU3(nn.Module):
    def __init__(self, num_classes, hidden_size=512, bidirectional=False, dropout=0.5, load_weight=None):
        super(OctResNetGRU3, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.resnet = oct_resnet50(pretrained=False, num_classes=48, zero_init_residual=True)
        if load_weight is not None:
            self.resnet.load_state_dict(torch.load(load_weight))
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

        if bidirectional:
            self.gru1 = nn.GRU(self.resnet.fc.in_features, hidden_size=hidden_size//2, batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(hidden_size, hidden_size=hidden_size//2, batch_first=True,bidirectional=True)
            self.gru3 = nn.GRU(hidden_size, hidden_size=hidden_size//2, batch_first=True,bidirectional=True)
        else:
            self.gru1 = nn.GRU(self.resnet.fc.in_features, hidden_size=hidden_size, batch_first=True, bidirectional=False)
            self.gru2 = nn.GRU(hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
            self.gru3 = nn.GRU(hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)

        self.fc = nn.Linear(hidden_size, num_classes)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x_h, x_l = self.layer1(x)
        x_h, x_l = self.layer2((x_h, x_l))
        x_h, x_l = self.layer3((x_h, x_l))
        x_h, x_l = self.layer4((x_h, x_l))
        x = self.avg_pool(x_h)
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
        x, _ = self.gru3(x, hidden)

        x = self.fc(x.contiguous().view(-1, x.size(2)))
        return x

    def forward(self, x):
        b, s, c, h, w = x.size()
        x = self.encode(x.view(-1, c, h, w))
        x = self.decode(x.view(b, s, -1))
        return x



def test():
    inputs = torch.FloatTensor(8, 3, 3, 224, 224)
    model = OctResNetGRU3(10, 512, bidirectional=True)
    logits = model(inputs)
    print(logits.size())

if __name__ == '__main__':
    test()
