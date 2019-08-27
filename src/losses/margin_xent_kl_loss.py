import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginCrossEntropyKLLoss(nn.CrossEntropyLoss):
    def __init__(self, alpha=1.0, beta=0.2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(MarginCrossEntropyKLLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        probs = F.softmax(input, 1)
        negprobs = probs.clone()
        negprobs[torch.arange(len(target)), target] = -1
        neg = negprobs.argmax(dim=1)
        poslogprobs = torch.clamp(torch.log(probs), -10000, 0)
        neglogprobs = torch.clamp(torch.log(1-negprobs), -10000, 0)

        pos_xent_loss = F.nll_loss(poslogprobs, target, self.weight, None, self.ignore_index, None, self.reduction)
        neg_xent_loss = self.alpha * F.nll_loss(neglogprobs, neg, self.weight, None, self.ignore_index, None, self.reduction)
        xent_loss = pos_xent_loss + self.alpha * neg_xent_loss

        logprobs = F.log_softmax(input, 1)
        onehot = torch.zeros((input.size(0), input.size(1)), dtype=input.dtype, device=input.device)
        onehot.scatter_(1, target.unsqueeze(dim=1), 1)
        kl_loss = F.kl_div(logprobs, onehot, reduction='batchmean')
        return (1-self.beta)*xent_loss + self.beta * kl_loss


if __name__ == '__main__':
    x = torch.FloatTensor([[-5e+2, 1e+3, 3e-1], [2e-4, -3e+2, 3e+3]])
    # y = torch.tensor([1, 2])
    y = torch.tensor([2, 2])
    criterion = MarginCrossEntropyKLLoss(alpha=1.0, beta=0.2)
    print(criterion(x, y))
