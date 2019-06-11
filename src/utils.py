import os
import json
import pandas as pd

def get_vocab():
    with open('../input/vocab/char2unicode.json', 'r') as f:
        char2uni = json.load(f)
    with open('../input/vocab/unicode2char.json', 'r') as f:
        uni2char = json.load(f)
    with open('../input/vocab/index2unicode.json', 'r') as f:
        index2uni = json.load(f)
    with open('../input/vocab/unicode2index.json', 'r') as f:
        uni2index = json.load(f)

    return {'char2uni': char2uni,
            'uni2char': uni2char,
            'index2uni': index2uni,
            'uni2index': uni2index}


def get_train_df():
    return pd.read_csv('../input/tables/train.csv')

def get_test_df():
    return pd.read_csv('../input/tables/train.csv')

def get_char_df():
    return pd.read_csv('../input/tables/character.csv')

def set_seed(seed):
    pass