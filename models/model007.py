import torch
import torch.nn as nn
from .abn_resnet import *

class ABN_ResNetLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=512, bidirectional=False, dropout=0.5, load_weight=None):
        super(ABN_ResNetLSTM, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.resnet = abn_resnet50(pretrained=True, num_classes=num_classes)
        if load_weight is not None:
            self.resnet.load_state_dict(torch.load(load_weight))
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3

        self.att_layer4 = self.resnet.att_layer4
        self.bn_att = self.resnet.bn_att
        self.att_conv = self.resnet.att_conv
        self.bn_att2 = self.resnet.bn_att2
        self.att_conv3 = self.resnet.att_conv3
        self.bn_att3 = self.resnet.bn_att3
        self.sigmoid = self.resnet.sigmoid
        self.att_conv2 = self.resnet.att_conv2
        self.att_gap = self.resnet.att_gap

        self.layer4 = self.resnet.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

        if bidirectional:
            self.lstm = nn.LSTM(self.resnet.fc.in_features, hidden_size=hidden_size//2, batch_first=True,bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.resnet.fc.in_features, hidden_size=hidden_size, batch_first=True, bidirectional=False)

        self.fc = nn.Linear(hidden_size, num_classes)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        ax = self.bn_att(self.att_layer4(x))
        ax = self.relu(self.bn_att2(self.att_conv(ax)))
        bs, cs, ys, xs = ax.shape
        self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))
        ax = self.att_conv2(ax)
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)

        rx = x * self.att
        rx = rx + x
        rx = self.layer4(rx)
        rx = self.avg_pool(rx)
        if self.dropout is not None:
            rx = self.dropout(rx)
        rx = rx.view(x.size(0), -1)

        return rx, ax

    def decode(self, x):
        batch, seq, _ = x.size()
        if self.bidirectional:
            hidden = (torch.zeros(2, batch, self.hidden_size//2, dtype=x.dtype, device=x.device),
                        torch.zeros(2, batch, self.hidden_size//2, dtype=x.dtype, device=x.device))
        else:
            hidden = (torch.zeros(1, batch, self.hidden_size, dtype=x.dtype, device=x.device),
                        torch.zeros(1, batch, self.hidden_size, dtype=x.dtype, device=x.device))

        x, _ = self.lstm(x, hidden)
        x = self.fc(x.contiguous().view(-1, x.size(2)))
        return x

    def forward(self, x):
        b, s, c, h, w = x.size()
        rx, ax = self.encode(x.view(-1, c, h, w))
        rx = self.decode(rx.view(b, s, -1))
        return rx, ax, self.att



def test():
    inputs = torch.FloatTensor(8, 3, 3, 224, 224)
    model = ABN_ResNetLSTM(10, 512, bidirectional=True)
    # logits = model(inputs)
    rx, ax, att = model(inputs)
    print(rx.size())
    print(ax.size())
    print(att.size())

if __name__ == '__main__':
    test()
