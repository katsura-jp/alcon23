import torch
import torch.nn as nn
import torch.nn.functional as F
'''
posのXENTとnegの最大のXENT
posのLossを最小化、negを最大化したい
log(softmax(pos)) + log(1-softmax(neg))
'''

class MarginCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, alpha=1.0, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(MarginCrossEntropy, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.alpha = alpha
    def forward(self, input, target):
        probs = F.softmax(input, 1)
        negprobs = probs.clone()
        negprobs[torch.arange(len(target)), target] = -1
        neg = negprobs.argmax(dim=1)
        poslogprobs = torch.clamp(torch.log(probs), -10000, 0)
        neglogprobs = torch.clamp(torch.log(1-negprobs), -10000, 0)
        return F.nll_loss(poslogprobs, target, self.weight, None, self.ignore_index, None, self.reduction) + \
               self.alpha * F.nll_loss(neglogprobs, neg, self.weight, None, self.ignore_index, None, self.reduction)



if __name__ == '__main__':
    x = torch.FloatTensor([[-5e+2, 1e+3, 3e-1], [2e-4, -3e+2, 3e+3]])
    # y = torch.tensor([1, 2])
    y = torch.tensor([2, 2])
    criterion = MarginCrossEntropy()
    print(criterion(x, y))
