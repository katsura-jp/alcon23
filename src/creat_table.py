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

import utils

#TODO: joblib or multiprocess

def main():
    os.makedirs('../input/tables', exist_ok=True)
    print('=== make train table ===')
    creat_train_table()
    print('=== make  test table ===')
    creat_test_table()
    print('=== make character table ===')
    creat_character_table()

def creat_train_table():
    # 配布されたcsvファイルを読み込む
    train_df = pd.read_csv('../input/dataset/train/annotations.csv')
    

    vocab = utils.get_vocab()
    with open('../input/vocab/rarity.json', 'r') as f:
        rarity = json.load(f)


    train_dict = {
        'ID' : list(),
        'file': list(),
        'Unicode1': list(),
        'Unicode2': list(),
        'Unicode3': list(),
        'target1': list(),
        'target2': list(),
        'target3': list(),
        'label' : list(),
        'rarity': list(),
        'split1': list(),
        'split2': list(),
        'margin': list(),
        'height' : list(),
        'width': list(),
        'aspect': list()
    }

    for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
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

        train_dict['ID'].append(ID)
        train_dict['file'].append(filename)

        train_dict['Unicode1'].append(uni1)
        train_dict['Unicode2'].append(uni2)
        train_dict['Unicode3'].append(uni3)
        
        train_dict['target1'].append(target1)
        train_dict['target2'].append(target2)
        train_dict['target3'].append(target3)
        
        train_dict['label'].append(label)

        train_dict['height'].append(h)
        train_dict['width'].append(w)
        train_dict['aspect'].append(aspect)

        train_dict['rarity'].append(rare)
        
        train_dict['split1'].append(split1)
        train_dict['split2'].append(split2)
        train_dict['margin'].append(margin)


    skf = StratVifiedKFold(n_splits=5)
    split = np.zeros(len(train_dict["ID"]))
    for k, (train_index, val_index) in enumerate(skf.split(range(len(train_dict["ID"])), train_dict['rarity'])):
        split[val_index] = k

    train_dict['split'] = split
    
    meta = pd.DataFrame.from_dict(train_dict)
    meta = meta.sort_values('ID').set_index('ID')
    meta.to_csv('../input/tables/meta-train.csv')
    drop_columns = ['height', 'width', 'aspect', 'rarity']
    meta.drop(drop_columns, axis=1).to_csv('../input/tables/train.csv')
        


def creat_test_table():
    # 配布されたcsvファイルを読み込む
    test_df = pd.read_csv('../input/dataset/test/annotations.csv')
    test_dict = {
        'ID' : list(),
        'file' : list(),
        'split1': list(),
        'split2': list(),
        'margin': list(),
        'height': list(),
        'width': list(),
        'aspect': list()
    }

    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        ID = row['ID']
        filename = str(ID) + '.jpg'
        image = np.array(Image.open(os.path.join('../input/dataset/test/imgs/', filename)))
        h, w, c = image.shape
        aspect = h / w
        split1, split2, margin = get_split_point(image)

        test_dict['ID'].append(ID)
        test_dict['file'].append(filename)
        test_dict['split1'].append(split1)
        test_dict['split2'].append(split2)
        test_dict['margin'].append(margin)
        test_dict['height'].append(h)
        test_dict['width'].append(w)
        test_dict['aspect'].append(aspect)

    
    meta = pd.DataFrame.from_dict(test_dict)
    meta = meta.sort_values('ID').set_index('ID')
    meta.to_csv('../input/tables/meta-test.csv')
    drop_columns = ['height', 'width', 'aspect']
    meta.drop(drop_columns, axis=1).to_csv('../input/tables/test.csv')



def creat_character_table():
    
    vocab = utils.get_vocab()
    with open('../input/vocab/rarity.json', 'r') as f:
        rarity = json.load(f)
    
    char_dict = {
        'file': list(),
        'Unicode': list(),
        'target': list(),
        'label': list(),
        'rarity': list(),
        'height': list(),
        'width': list(),
        'aspect': list(),
        }

    for uni in tqdm(vocab['index2uni']):
        image_paths = glob.glob(os.path.join('../input/dataset/kana_train/', uni, '*.jpg'))
        rare = rarity[uni]
        label = vocab['uni2char'][uni]
        target = vocab['uni2index'][uni]

        for path in image_paths:
            image = np.array(Image.open(path))
            filename = path.split('/')[-1]
            h, w, c = image.shape
            aspect = h / w
            char_dict['file'].append(filename)
            char_dict['Unicode'].append(uni)
            char_dict['target'].append(target)
            char_dict['label'].append(label)
            char_dict['rarity'].append(rare)
            char_dict['height'].append(h)
            char_dict['width'].append(w)
            char_dict['aspect'].append(aspect)
        
    skf = StratVifiedKFold(n_splits=5)
    split = np.zeros(len(train_dict["ID"]))
    for k, (train_index, val_index) in enumerate(skf.split(char_dict["file"], char_dict['rarity'])):
        split[val_index] = k

    char_dict['split'] = split
    
    meta = pd.DataFrame.from_dict(char_dict)
    meta = meta.sort_values('target').set_index('file')
    meta.to_csv('../input/tables/meta-character.csv')
    drop_columns = ['height', 'width', 'aspect', 'rarity']
    meta.drop(drop_columns, axis=1).to_csv('../input/tables/character.csv')


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
