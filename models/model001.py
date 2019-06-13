import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
__all__ = ['EncoderDecoderResNet']

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        # x = x.unsqueeze(1)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, in_features, hidden_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)
        # self.gru = nn.GRU(in_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x, hidden


class EncoderDecoderResNet(nn.Module):
    def __init__(self, num_classes, hidden_dim=512):
        super(EncoderDecoderResNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes, self.encoder.output_dim,hidden_dim)
        self.hidden_dim = hidden_dim
    def forward(self, x):
        '''
        :param x: torch.tensor(batch, 3(char), 3(channel), hight, width)
        :return: torch.tensor(batch, 3(char), num_classes)
        '''
        bs, char, c, h, w = x.size()
        x = x.view(bs*char, c, h, w) #batch*char x 3 x height x width
        x = self.encoder(x) # (batch * char) x 1 x output_dim
        x = x.view(bs, char, -1)
        hidden = (torch.zeros(1, bs, self.hidden_dim, dtype=x.dtype, device=x.device),
                  torch.zeros(1, bs, self.hidden_dim, dtype=x.dtype, device=x.device))

        output = []
        for i in range(3):
            out, hidden = self.decoder(x[:, i], hidden)
            output.append(out)
        output = torch.stack(output, dim=0).transpose(0, 1)

        return output


def test():
    inputs = torch.zeros(4, 3, 3, 128, 128)
    model = EncoderDecoderResNet(10, hidden_dim=512)
    logits = model(inputs)
    print(logits.size())
    print('success')


if __name__ == '__main__':
    test()