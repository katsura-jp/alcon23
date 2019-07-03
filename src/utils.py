import os
import json
import pandas as pd
import torch
import numpy as np
import random


def get_vocab(path='../input/vocab'):
    with open(os.path.join(path, 'char2unicode.json'), 'r') as f:
        char2uni = json.load(f)
    with open(os.path.join(path, 'unicode2char.json'), 'r') as f:
        uni2char = json.load(f)
    with open(os.path.join(path, 'index2unicode.json'), 'r') as f:
        index2uni = json.load(f)
    with open(os.path.join(path, 'unicode2index.json'), 'r') as f:
        uni2index = json.load(f)

    return {'char2uni': char2uni,
            'uni2char': uni2char,
            'index2uni': index2uni,
            'uni2index': uni2index}


def get_train_df(path='../input/tables'):
    return pd.read_csv(os.path.join(path, 'train.csv'))

def get_test_df(path='../input/tables'):
    return pd.read_csv(os.path.join(path, 'test.csv'))

def get_char_df(path='../input/tables'):
    return pd.read_csv(os.path.join(path, 'character.csv'))

def seed_setting(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_resolution(n=1):
    if n == 1:
        return 336, 224, 384, 256
    elif n == 2:
        return 168 , 112  ,192 ,128
    elif n == 3:
        return 128 , 128 ,150, 150
    elif n == 4:
        return 84, 56, 96, 64
    elif n == 5:
        return 64, 64 , 72 ,72
    elif n == 6:
        return 192, 128, 210, 140


def mixup_data(x, y, alpha=0.2, device="cpu"):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
