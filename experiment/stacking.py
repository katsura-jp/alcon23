import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import yaml
import datetime
import logging
import math


import torch.nn as nn
from torch.utils.data import DataLoader
import tensorboardX as tbx
from fastprogress import progress_bar, master_bar
import apex.amp as amp

sys.path.append('../')
from src.stacking_dataset import StackingDataset
from models import *
from src.augmentation import get_test_augmentation, get_train_augmentation
from src.dataset import AlconDataset, KanaDataset
from src.metrics import *
from src.utils import *
from src.collates import *
from src.scheduler import *
from src.losses import *
from train_methods import *

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(240, 256)
        self.fc2 = nn.Linear(256, 48)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # x: bs x 3 x 240
        bs = x.shape[0]
        x = x.view(-1, 240) # bs*3 x 240
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # bs*3 x 48
        return x

EXP_NAME = 'stacking'

def main():
    n_epoch = 10

    now = datetime.datetime.now()
    now_date = '{}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

    # set logger
    logger = logging.getLogger("Log")
    logger.setLevel(logging.DEBUG)

    handler_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    print('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    with open('../params/stacking.yaml', "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)

    seed_setting(param['seed'])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    local_cv = dict()

    for fold in range(5):
        outdir = os.path.join(param['save path'], EXP_NAME, now_date, 'fold{}'.format(fold))
        if os.path.exists(param['save path']):
            os.makedirs(outdir, exist_ok=True)
        else:
            print("Not find {}".format(param['save path']))
            raise FileNotFoundError

        file_handler = logging.FileHandler(os.path.join(outdir, 'experiment.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(handler_format)
        logger.addHandler(file_handler)

        logger.debug('=============   FOLD  {}  ============='.format(fold))
        logger.debug('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))

        print(f'fold - {fold}')
        print('load data set')
        train_dataset = StackingDataset(df=get_train_df(param['tabledir']).query('valid != @fold'),
                                        logit_path='/mnt/hdd1/alcon2019/logits_for_oof.pth',
                                        mode='train')
        valid_dataset = StackingDataset(df=get_train_df(param['tabledir']).query('valid == @fold'),
                                        logit_path='/mnt/hdd1/alcon2019/logits_for_oof.pth',
                                        mode='valid')

        print('load data loader')
        train_dataloader = DataLoader(train_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False, shuffle=False)

        print('model set')
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model = model.to(param['device'])
        loss_fn = nn.CrossEntropyLoss().to(param['device'])
        eval_fn = accuracy_one_character

        max_char_acc = -1e-5
        max_3char_acc = -1e-5
        min_loss = 1e+5

        mb = master_bar(range(n_epoch))

        for epoch in mb:
            model.train()
            avg_train_loss = 10**5
            avg_train_accuracy = 0.0
            avg_three_train_acc = 0.0
            for step, (inputs, targets, indice) in enumerate(progress_bar(train_dataloader, parent=mb)):
                model.train()
                inputs = inputs.to(param['device'])
                targets = targets.to(param['device'])
                optimizer.zero_grad()
                logits = model(inputs)  # logits.size() = (batch*3, 48)
                preds = logits.view(targets.size(0), 3, -1).softmax(dim=2)
                loss = loss_fn(logits, targets.view(-1, targets.size(2)).argmax(dim=1))
                loss.backward()
                avg_train_loss += loss.item()
                _avg_accuracy = eval_fn(preds, targets.argmax(dim=2)).item()
                avg_train_accuracy += _avg_accuracy
                _three_char_accuracy = accuracy_three_character(preds, targets.argmax(dim=2), mean=True).item()
                avg_three_train_acc += _three_char_accuracy

            avg_train_loss /= len(train_dataloader)
            avg_train_accuracy /= len(train_dataloader)
            avg_three_train_acc /= len(train_dataloader)

            avg_valid_loss, avg_valid_accuracy, avg_three_valid_acc = valid_alcon_rnn(model, valid_dataloader, param['device'],
                                                                                  loss_fn, eval_fn)

            if min_loss > avg_valid_loss:
                logger.debug('update best loss:  {:.5f} ---> {:.5f}'.format(min_loss, avg_valid_loss))
                min_loss = avg_valid_loss
                torch.save(model.state_dict(), os.path.join(outdir, 'best_loss.pth'))

            if max_char_acc < avg_valid_accuracy:
                logger.debug('update best acc per 1 char:  {:.3%} ---> {:.3%}'.format(max_char_acc, avg_valid_accuracy))
                max_char_acc = avg_valid_accuracy
                torch.save(model.state_dict(), os.path.join(outdir, 'best_acc.pth'))

            if max_3char_acc < avg_three_valid_acc:
                logger.debug(
                    'update best acc per 3 char:  {:.3%} ---> {:.3%}'.format(max_3char_acc, avg_three_valid_acc))
                max_3char_acc = avg_three_valid_acc
                torch.save(model.state_dict(), os.path.join(outdir, 'best_3acc.pth'))

            logger.debug('======================== epoch {} ========================'.format(epoch+1))
            logger.debug('lr              : {:.5f}'.format(scheduler.get_lr()[0]))
            logger.debug('loss            : train={:.5f}  , test={:.5f}'.format(avg_train_loss, avg_valid_loss))
            logger.debug('acc(per 1 char) : train={:.3%}  , test={:.3%}'.format(avg_train_accuracy, avg_valid_accuracy))
            logger.debug('acc(per 3 char) : train={:.3%}  , test={:.3%}'.format(avg_three_train_acc, avg_three_valid_acc))

        logger.debug('================  FINISH  TRAIN  ================')
        logger.debug('Result')
        logger.debug('Best loss : {}'.format(min_loss))
        logger.debug('Best 1 acc : {}'.format(max_char_acc))
        logger.debug('Best 3 acc : {}'.format(max_3char_acc))





if __name__ == '__main__':
    main()