import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothLoss(nn.Module):
    def __init__(self, alpha=1.0, reduction='mean'):
        super(LabelSmoothLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        logprobs = F.log_softmax(input, 1)
        print(F.kl_div(logprobs, target.type(torch.float), reduction=self.reduction))
        return self.alpha * F.kl_div(logprobs, target.type(torch.float), reduction=self.reduction) + \
                (1 - self.alpha) * F.nll_loss(logprobs, target.argmax(dim=1), None, None, -100, None, self.reduction)
if __name__ == '__main__':
    x = torch.FloatTensor([[-5e+2, 1e+3, 3e-1],[2e-4, -3e+2, 3e+3]])
    y = torch.tensor([1, 2])
    y = torch.tensor([[0,1,0],[0,0,1]])
    criterion = LabelSmoothLoss(alpha=0.2)
    print(criterion(x, y))
    x = torch.FloatTensor(2,3)
    print(criterion(x, y))