import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, gamma, weights, mean=False):
        super(TverskyLoss, self).__init__()
        self.gamma = gamma
        self.weights = torch.tensor(weights).cuda()
        self.mean = mean

    def forward(self, logits, true, alpha=0.7, beta=0.3, eps=1e-7):
        """Computes the Tversky loss [1].
        Args:

        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.

        true: a tensor of shape [B, H, W] or [B, 1, H, W].

        alpha: controls the penalty for false positives.

        beta: controls the penalty for false negatives.

        eps: added to the denominator for numerical stability.

        Returns:

        tversky_loss: the Tversky loss.

        Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff

        References:

        [1]: https://arxiv.org/abs/1706.05721
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.long().squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)

        else:

            true_1_hot = true
            probas = F.softmax(logits, dim=1)

        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + alpha * fps + beta * fns

        tversky_loss = (num + eps + 1) / (denom + eps + 1)

        if len(self.weights.size()) != 0:
            if self.mean:
                return (((1 - tversky_loss) ** (self.gamma)) * self.weights).mean()
            else:
                return (((1 - tversky_loss) ** (self.gamma)) * self.weights).sum()


        elif len(self.weights.size()) == 0:
            return ((1 - tversky_loss) ** (self.gamma)).mean()