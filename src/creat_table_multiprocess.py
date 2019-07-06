import os
import sys
import glob

import numpy as np
import pandas as pd
import PIL.Image as Image
from scipy.stats import norm
from tqdm import tqdm
import cv2
from sklearn.model_selection import StratifiedKFold
import json
from joblib import Parallel, delayed

import utils


def main():
    os.makedirs('../input/tables', exist_ok=True)
    print('=== make train table ===')
    creat_train_table()
    print('=== make  test table ===')
    creat_test_table()
    print('=== make character table ===')
    creat_character_table()

def creat_train_table(seed=531):
    # 配布されたcsvファイルを読み込む
    train_df = pd.read_csv('../input/dataset/train/annotations.csv')
    
    vocab = utils.get_vocab()
    with open('../input/vocab/rarity.json', 'r') as f:
        rarity = json.load(f)


    train_list = Parallel(n_jobs = -1)([delayed(process_train)(row, vocab, rarity) for index, row in tqdm(train_df.iterrows(), total=len(train_df))])

    meta = pd.DataFrame(train_list).sort_values('ID')
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    
    for k, (train_index, val_index) in enumerate(skf.split(meta.index, meta.rarity)):
        meta.loc[val_index, 'valid'] = k
    meta = meta.set_index('ID')
    meta.to_csv('../input/tables/meta-train.csv')
    drop_columns = ['height', 'width', 'aspect', 'rarity']
    meta.drop(drop_columns, axis=1).to_csv('../input/tables/train.csv')

def process_train(row, vocab,rarity):
    train_dict = dict()
    ID = row['ID']
    filename = str(ID) +'.jpg'
    image = np.array(Image.open(os.path.join('../input/dataset/train/imgs', filename)))
    uni1 = row['Unicode1']
    uni2 = row['Unicode2']
    uni3 = row['Unicode3']

    target1 = vocab['uni2index'][uni1]
    target2 = vocab['uni2index'][uni2]
    target3 = vocab['uni2index'][uni3]

    h, w, c = image.shape
    aspect = h / w

    label = vocab['uni2char'][uni1] + \
            vocab['uni2char'][uni2] + \
            vocab['uni2char'][uni3]

    rare = max([rarity[uni1], rarity[uni2], rarity[uni3]])

    split1, split2, margin = get_split_point(image)

    train_dict['ID'] = ID
    train_dict['file'] = filename

    train_dict['Unicode1'] = uni1
    train_dict['Unicode2'] = uni2
    train_dict['Unicode3'] = uni3
    
    train_dict['target1'] = target1
    train_dict['target2'] = target2
    train_dict['target3'] = target3
    
    train_dict['label'] = label

    train_dict['height'] = h
    train_dict['width'] = w
    train_dict['aspect'] = aspect

    train_dict['rarity'] = rare
    
    train_dict['split1'] = split1
    train_dict['split2'] = split2
    train_dict['margin'] = margin
    train_dict['valid'] = -1
    return train_dict


def creat_test_table():
    # 配布されたcsvファイルを読み込む
    test_df = pd.read_csv('../input/dataset/test/annotations.csv')

    test_list = Parallel(n_jobs=-1)([delayed(process_test)(row) for index, row in tqdm(test_df.iterrows(), total=len(test_df))])
    meta = pd.DataFrame(test_list)
    meta = meta.sort_values('ID').set_index('ID')
    meta.to_csv('../input/tables/meta-test.csv')
    drop_columns = ['height', 'width', 'aspect']
    meta.drop(drop_columns, axis=1).to_csv('../input/tables/test.csv')

def process_test(row):
    test_dict = {}

    ID = int(row['ID'])
    filename = str(ID) + '.jpg'
    image = np.array(Image.open(os.path.join('../input/dataset/test/imgs/', filename)))
    h, w, c = image.shape
    aspect = h / w
    split1, split2, margin = get_split_point(image)

    test_dict['ID'] = ID
    test_dict['file'] = filename
    test_dict['split1'] = split1
    test_dict['split2'] = split2
    test_dict['margin'] = margin
    test_dict['height'] = h
    test_dict['width'] = w
    test_dict['aspect'] = aspect
    return test_dict



def creat_character_table(seed=531):
    
    vocab = utils.get_vocab()
    with open('../input/vocab/rarity.json', 'r') as f:
        rarity = json.load(f)

    image_paths = glob.glob(os.path.join('../input/dataset/train_kana/U+*/*.jpg'))
    char_list = Parallel(n_jobs=-1)([delayed(process_char)(path, vocab, rarity) for path in tqdm(image_paths, total=len(image_paths))])
    
    meta = pd.DataFrame(char_list).sort_values('target')
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True) #split変えても良さそう
    for k, (train_index, val_index) in enumerate(skf.split(meta.index, meta.rarity)):
        meta.loc[val_index, 'valid'] = k

    meta = meta.set_index('file')
    meta.to_csv('../input/tables/meta-character.csv')
    drop_columns = ['height', 'width', 'aspect', 'rarity']
    meta.drop(drop_columns, axis=1).to_csv('../input/tables/character.csv')

def process_char(path, vocab, rarity):
    char_dict = dict()
    uni = path.split('/')[-2]
    rare = rarity[uni]
    label = vocab['uni2char'][uni]
    target = vocab['uni2index'][uni]
    
    image = np.array(Image.open(path))
    filename = path.split('/')[-1]
    h, w, c = image.shape
    aspect = h / w
    char_dict['file'] = filename
    char_dict['Unicode'] = uni
    char_dict['target'] = target
    char_dict['label'] = label
    char_dict['rarity'] = rare
    char_dict['height'] = h
    char_dict['width'] = w
    char_dict['aspect'] = aspect
    char_dict['valid'] = -1
    return char_dict


def get_split_point(image, mode='mean', alpha=0.7, gamma = 6):
    margin = int(image.shape[0] * 0.1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    x = [i for i in range(image.shape[0])]
    th2 = 1-th2/255. + alpha
    if mode == 'mean':
        w = th2.mean(axis=1)
    elif mode == 'sum':
        w = th2.sum(axis=1)
    else:
        raise NotImplementedError

    sp = int(image.shape[0] / 3)
    w1 = np.array(w[:sp * 2]) * np.array([(1-norm.pdf(x=i, loc=0, scale=1))**gamma for i in np.linspace(-1.0, 1.0, len(w[:sp*2]))])
    w2 = np.array(w[sp:]) * np.array([(1-norm.pdf(x=i, loc=0, scale=1))**gamma for i in np.linspace(-1.0, 1.0, len(w[sp:]))])

    split1 = w1.argmin()
    split2 = w2.argmin() + sp
    return split1, split2, margin


if __name__ == '__main__':
    main()    
