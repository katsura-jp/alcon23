import torch
import torch.nn as nn
from torchvision.models import densenet201

class DenseNet201GRU2(nn.Module):
    def __init__(self, num_classes, hidden_size=512, bidirectional=False, dropout=0.5, load_weight=None):
        super(DenseNet201GRU2, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.backbone = densenet201(pretrained=True)
        self.features = self.backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

        if bidirectional:
            self.gru1 = nn.GRU(self.backbone.classifier.in_features, hidden_size=hidden_size//2, batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(hidden_size, hidden_size=hidden_size//2, batch_first=True,bidirectional=True)
        else:
            self.gru1 = nn.GRU(self.resnet.fc.in_features, hidden_size=hidden_size, batch_first=True, bidirectional=False)
            self.gru2 = nn.GRU(hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def encode(self, x):
        x = self.features(x)
        x = self.relu(x)
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
    # inputs = torch.FloatTensor(6, 3, 3,200, 200)
    inputs = torch.FloatTensor(16, 3, 3, 192 , 128)
    model = DenseNet201GRU2(48, 512, bidirectional=True)
    # model = model.to('cuda:0')
    # inputs = inputs.to('cuda:0')
    logits = model(inputs)
    print(logits.size())

if __name__ == '__main__':
    test()
