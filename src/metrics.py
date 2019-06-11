import torch
import numpy as np

def accuracy(pred, label):
    '''
    param:
        pred : B x 3 x class
        label: B x 3 x 1
    '''
    acc = 0.0

    for batch in range(pred.size(0)):
        if pred[batch] == label:
            acc += 1.0

    acc /= pred.size(0)
    return acc


def accuracy_three_character(pred, label, mean=True):
    '''
    :param pred: torch.tensor(batch_size, 3, class)
    :param label: torch.tensor(batch_size, 3)
    :return: accuracy
    '''
    acc = torch.ones(pred.size(0), dtype='float32')
    for batch in range(pred.size(0)):
        for i in range(3):
            if pred[batch, i].argmax() != label[batch, i]:
                acc[batch] = 0.0
                break
    if mean:
        return acc / pred.size(0)
    else:
        return acc