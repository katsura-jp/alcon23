import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=0.2,reduction='mean'):
        super(LabelSmoothLoss, self).__init__()
        self.alpha = alpha
        self.conf = 1.0 - alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        '''
        :param input: torch.FloatTensor(batchsize, num_classes)
        :param target: torch.LongTensor(batchsize)
        :return: $\alpha \times KLLoss + (1-\alpha) \times CrossEntropyLoss$
        '''
        logprobs = F.log_softmax(input, 1)
        onehot = torch.zeros((input.size(0), input.size(1)), dtype=input.dtype, device=input.device)
        onehot.fill_(self.alpha)
        onehot.scatter_(1, target.unsqueeze(dim=1), self.conf)
        # onehot.masked_fill_((target == self.ignnore_index).unsqueeze(1), 0)

        kl_loss = F.kl_div(logprobs, onehot, reduction='batchmean') / input.size(1)
        # print(kl_loss)
        xent_loss = F.nll_loss(logprobs, target, None, None, -100, None, self.reduction)
        # print(xent_loss)
        return xent_loss + self.gamma*kl_loss

if __name__ == '__main__':
    x = torch.FloatTensor([[-5e+2, 1e+3, 3e-1],[2e-4, -3e+2, 3e+3]])
    y = torch.tensor([1, 2])
    # y = torch.tensor([[0,1,0],[0,0,1]])
    criterion = LabelSmoothLoss(alpha=0.2)
    print(criterion(x, y))
    x = torch.FloatTensor(2,3)
    print(criterion(x, y))
