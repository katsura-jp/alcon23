import os
import sys
sys.path.append('../')
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


from models import *
from src.augmentation import get_test_augmentation, get_train_augmentation
from src.dataset import AlconDataset, KanaDataset
from src.metrics import *
from src.utils import *
from src.collates import *
from src.scheduler import *
from src.losses import *
from train_methods import *

##############################
###  ATTENTION
##############################
# Change Param
# 1. EXP_NO & EXP_NAME
# 2. Model
# 3. Fold
# 4. Out dirctory
# 5. 
#
##############################
EXP_NO = 8
EXP_NAME = 'exp8'

def main():
    now = datetime.datetime.now()
    now_date = '{}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    
    # set logger
    logger = logging.getLogger("Log")
    logger.setLevel(logging.DEBUG)

    handler_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


    print('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    with open('../params/exp{}.yaml'.format(EXP_NO), "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    param['date'] = now_date
    # seed set
    seed_setting(param['seed'])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    local_cv = dict()

    for fold in [0]:
        # /mnt/hdd1/alcon2019/ + exp0/ + 2019-mm-dd_hh-mm-ss/ + foldN
        outdir = '/mnt/hdd1/alcon2019/exp8/2019-07-31_01-30-12/fold0/'

        file_handler = logging.FileHandler(os.path.join(outdir, 'experiment.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(handler_format)
        logger.addHandler(file_handler)



        # Dataset

        param['batch size'] = max(param['batch size'], param['batch size'] * param['GPU'])
        if param['debug']:
            train_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid != @fold').iloc[:param['batch size']*12],
                                         augmentation=get_train_augmentation(*get_resolution(param['resolution'])),
                                         datadir=os.path.join(param['dataroot'],'train','imgs'), mode='train')

            valid_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid == @fold').iloc[:param['batch size']*12],
                                         augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                         datadir=os.path.join(param['dataroot'],'train','imgs'), mode='valid')
        else:
            train_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid != @fold'),
                                         augmentation=get_train_augmentation(*get_resolution(param['resolution'])),
                                         datadir=os.path.join(param['dataroot'], 'train', 'imgs'), mode='train',
                                         margin_augmentation=True)

            valid_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid == @fold'),
                                         augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                         datadir=os.path.join(param['dataroot'], 'train', 'imgs'), mode='valid',
                                         margin_augmentation=False)

        # Dataloader


        train_dataloader = DataLoader(train_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False, shuffle=False)

        # model
        model = DenseNet201GRU2(num_classes=48, hidden_size=512, bidirectional=True, load_weight=None, dropout=param['dropout'])



        param['model'] = model.__class__.__name__

        # optim
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                        weight_decay=1e-5, nesterov=False)
        # scheduler


        model = model.to(param['device'])
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        if param['GPU'] > 0:
            model = nn.DataParallel(model)

        loss_fn = nn.CrossEntropyLoss().to(param['device'])
        eval_fn = accuracy_one_character

        max_char_acc = -1.
        max_3char_acc = -1.
        min_loss = 10**5



        max_char_acc = -1e-5
        max_3char_acc = -1e-5
        min_loss = 1e+5

        snapshot=0
        snapshot_loss_list = list()
        snapshot_eval_list = list()
        snapshot_eval3_list = list()
        snapshot_loss = 1e+5
        snapshot_eval = -1e-5
        snapshot_eval3 = -1e-5
        val_iter = math.ceil(len(train_dataloader) / 3)
        print('val_iter: {}'.format(val_iter))
        # Hyper params
        cycle_iter = 5
        snap_start = 2
        n_snap = 8

        mb = master_bar(range((n_snap+snap_start) * cycle_iter))
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=len(train_dataloader) * cycle_iter, T_mult=1, T_up=500,
                                                  eta_max=0.1)
        # local CV

        target_list = list()
        for _, targets, _ in valid_dataloader:
            targets = targets.argmax(dim=2)
            target_list.append(targets)
        target_list = torch.cat(target_list)

        mb = master_bar(range(n_snap))
        valid_logit_dict = dict()
        init = True
        for i in mb:
            print('load weight  :  {}'.format(os.path.join(outdir, f'best_loss_{i+1}.pth')))
            model.load_state_dict(torch.load(os.path.join(outdir, f'best_loss_{i+1}.pth')))
            logit_alcon_rnn(model, valid_dataloader, param['device'], valid_logit_dict, div=n_snap, init=init)
            init = False

        pred_list = torch.stack(list(valid_logit_dict.values()))
        pred_list = pred_list.softmax(dim=2)
        local_accuracy = accuracy_three_character(pred_list, target_list)
        logger.debug('LOCAL CV : {:5%}'.format(local_accuracy))
        torch.save(valid_logit_dict, os.path.join(outdir, f'fold{fold}_valid_logit.pth'))

        local_cv['fold{}'.format(fold)] = {'accuracy': local_accuracy, 'valid_size': len(valid_dataset)}

        del train_dataset, valid_dataset
        del train_dataloader, valid_dataloader
        del scheduler, optimizer
        del valid_logit_dict, target_list
        gc.collect()

        logger.debug('=========== Prediction phrase ===========')

        if param['debug']:
            test_dataset = AlconDataset(df=get_test_df(param['tabledir']).iloc[:param['batch size'] * 12],
                                        augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                        datadir=os.path.join(param['dataroot'], 'test', 'imgs'), mode='test')
        else:
            test_dataset = AlconDataset(df=get_test_df(param['tabledir']),
                                        augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                        datadir=os.path.join(param['dataroot'], 'test', 'imgs'), mode='test')

        test_dataloader = DataLoader(test_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                     pin_memory=False, drop_last=False, shuffle=False)
        logger.debug('test dataset size: {}'.format(len(test_dataset)))
        logger.debug('test loader size: {}'.format(len(test_dataloader)))

        test_logit_dict = dict()
        init = True
        for i in range(n_snap):
            print('load weight  :  {}'.format(os.path.join(outdir, f'best_loss_{i+1}.pth')))
            model.load_state_dict(torch.load(os.path.join(outdir, f'best_loss_{i+1}.pth')))
            logit_alcon_rnn(model, test_dataloader, param['device'], test_logit_dict, div=n_snap, init=init)
            init = False

        torch.save(test_logit_dict, os.path.join(outdir, 'prediction.pth'))
        output_list = make_submission(test_logit_dict)
        pd.DataFrame(output_list).sort_values('ID').set_index('ID').to_csv(os.path.join(outdir, 'test_prediction.csv'))
        logger.debug('success!')
        logger.removeHandler(file_handler)

        del test_dataset, test_dataloader
        gc.collect()

    print('success!')


if __name__ =='__main__':
    main()
