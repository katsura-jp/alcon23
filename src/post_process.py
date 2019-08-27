import torch
import numpy as np
import os
import math
import pandas as pd
from .utils import *
import sys
sys.path.append('../')
from models import *


def get_root_dir(n=7):
    if n == 7:
        rootdirs = ['/mnt/hdd1/alcon2019/exp7/2019-07-13_12-25-44/fold0/',
                    '/mnt/hdd1/alcon2019/exp7/2019-07-13_12-25-44/fold1/',
                    '/mnt/hdd1/alcon2019/exp7/2019-07-18_10-17-48/fold2/',
                    '/mnt/hdd1/alcon2019/exp7/2019-07-22_13-55-51/fold3/',
                    '/mnt/hdd1/alcon2019/exp7/2019-07-22_13-55-51/fold4/']
    elif n == 8:
        rootdirs = ['/mnt/hdd1/alcon2019/exp8/2019-07-31_01-30-12/fold0/',
                    '/mnt/hdd1/alcon2019/exp8/2019-08-01_04-55-40/fold1/',
                    '/mnt/hdd1/alcon2019/exp8/2019-08-01_04-55-40/fold2/',
                    '/mnt/hdd1/alcon2019/exp8/2019-08-01_04-55-40/fold3/',
                    '/mnt/hdd1/alcon2019/exp8/2019-08-01_04-55-40/fold4/']
    elif n == 9:
        rootdirs = ['/mnt/hdd1/alcon2019/exp9/2019-08-01_01-41-16/fold0/',
                    '/mnt/hdd1/alcon2019/exp9/2019-08-01_11-03-24/fold1/',
                    '/mnt/hdd1/alcon2019/exp9/2019-08-01_23-42-41/fold2/',
                    '/mnt/hdd1/alcon2019/exp9/2019-08-01_23-42-41/fold3/',
                    '/mnt/hdd1/alcon2019/exp9/2019-08-01_23-42-41/fold4/']
    elif n == 11:
        rootdirs = ['/mnt/hdd1/alcon2019/exp11/2019-08-03_01-34-03/fold0/',
                    '/mnt/hdd1/alcon2019/exp11/2019-08-03_01-34-03/fold1/',
                    '/mnt/hdd1/alcon2019/exp11/2019-08-03_01-34-03/fold2/',
                    '/mnt/hdd1/alcon2019/exp11/2019-08-03_01-34-03/fold3/',
                    '/mnt/hdd1/alcon2019/exp11/2019-08-03_01-34-03/fold4/']
    elif n == 16:
        rootdirs = ['/mnt/hdd1/alcon2019/exp16/2019-08-26_07-30-18/fold0/',
                    '/mnt/hdd1/alcon2019/exp16/2019-08-26_07-30-18/fold1/',
                    '/mnt/hdd1/alcon2019/exp16/2019-08-26_07-30-18/',
                    '/mnt/hdd1/alcon2019/exp16/2019-08-26_07-30-18/',
                    '/mnt/hdd1/alcon2019/exp16/2019-08-26_07-30-18/']
    else:
        raise "Not such file"
    
    return rootdirs


def get_model(n=7):
    if n == 7:
        model = OctResNetGRU2(num_classes=48, hidden_size=512, bidirectional=True, load_weight=None)
    elif n == 8:
        model = DenseNet201GRU2(num_classes=48, hidden_size=512, bidirectional=True, load_weight=None)
    elif n == 9:
        model = InceptionV4GRU2(num_classes=48, hidden_size=512, bidirectional=True, load_weight=None)
    elif n == 11:
        model = SEResNeXtGRU2(num_classes=48, hidden_size=512, bidirectional=True, load_weight=None)
    elif n == 16:
        model = InceptionV4GRU2(num_classes=48, hidden_size=512, bidirectional=True, load_weight=None)
    else:
        raise "Not such file"
    return model
    

def get_test_prediction(n=7):
    rootdirs = get_root_dir(n)
    n_fold = len(rootdirs)
    ensemble_predict = dict()
    for fold, rootdir in enumerate(rootdirs):
        path = os.path.join(rootdir, 'prediction.pth')
        fold_predict = torch.load(path)
        if fold == 0:
            for k, v in fold_predict.items():
                ensemble_predict[k] = v / n_fold
        else:
            for k, v in fold_predict.items():
                ensemble_predict[k] += v / n_fold
    return ensemble_predict

def get_valid_prediction(n=7):
    rootdirs = get_root_dir(n)
    n_fold = len(rootdirs)
    predict = dict()
    for fold, rootdir in enumerate(rootdirs):
        path = os.path.join(rootdir, f'fold{fold}_valid_logit.pth')
        part_preds = torch.load(path)
        for k, v in part_preds.items():
            predict[k] = v
    predict = sorted(predict.items())
    predict = dict((k, v) for k, v in predict)
    return predict
            

def make_submission(prediction : dict) -> list:
    '''
    :param prediction: dict {ID : logit}
    :return: list
    '''
    submission_list = list()
    vocab = get_vocab()
    for index, logit in prediction.items():
        pred_dict = dict()
        pred = logit.softmax(dim=1)
        pred_dict['ID'] = index
        pred_dict['Unicode1'] = vocab['index2uni'][int(pred[0].argmax(dim=0).item())]
        pred_dict['Unicode2'] = vocab['index2uni'][int(pred[1].argmax(dim=0).item())]
        pred_dict['Unicode3'] = vocab['index2uni'][int(pred[2].argmax(dim=0).item())]
        submission_list.append(pred_dict)

    return submission_list

def submission_to_df(submit_list : list) -> pd.DataFrame:
    return pd.DataFrame(submit_list).sort_values('ID').set_index('ID')
