import torch
import numpy as np

def accuracy(pred, label, mean=True):
    '''
    param:
        pred : B x 3 x class
        label: B x 3 x 1
    PyTorch
    '''
    if mean:
        return (pred.argmax(dim=1) == label).type(torch.float32).mean()
    else:
        return (pred.argmax(dim=1) == label)


def accuracy_one_character(pred, label, mean=True):
    if mean:
        return (pred.argmax(dim=2) == label).float().mean(dim=[0, 1])
    else:
        return (pred.argmax(dim=2) == label).float().mean(dim=1)


def accuracy_three_character(pred, label, mean=True):
    '''
    :param pred: torch.tensor(batch_size, 3, class)
    :param label: torch.tensor(batch_size, 3)
    :return: accuracy
    PyTorch
    '''
    if mean:
        return (pred.argmax(dim=2) == label).all(dim=1).type(torch.float32).mean()
    else:
        return (pred.argmax(dim=2) == label).all(dim=1)