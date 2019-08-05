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
from src import post_process
from train_methods import *

EXP_NO = None
EXP_NAME = None

def main():
    now = datetime.datetime.now()
    now_date = '{}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

    # set logger
    logger = logging.getLogger("Log")
    logger.setLevel(logging.DEBUG)

    handler_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


    print('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    with open('../params/prediction.yaml', "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)

    param['date'] = now_date
    EXP_NO = param['exp']
    EXP_NAME = f'exp{EXP_NO}'

    outdir = os.path.join(param['save path'], EXP_NAME ,now_date)

    if os.path.exists(param['save path']):
        os.makedirs(outdir, exist_ok=True)
    else:
        print("Not find {}".format(param['save path']))
        raise FileNotFoundError


    file_handler = logging.FileHandler(os.path.join(outdir, 'experiment.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)

    logger.debug('=============   Prediction  ============='.)
    logger.debug('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))


    # seed set
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    local_cv = dict()
    rootdirs = post_process.get_root_dirs(EXP_NO) 
    model = post_process.get_model(EXP_NO)
    # optim
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                    weight_decay=1e-5, nesterov=False)
    # scheduler


    model = model.to(param['device'])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if param['GPU'] > 0:
        model = nn.DataParallel(model)
    snap_range = param['snap_range']

    for fold in range(5):
        rootdir = rootdirs[fold]
        # Dataset
        param['batch size'] = max(param['batch size'], param['batch size'] * param['GPU'])

        valid_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid == @fold'),
                                     augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                     datadir=os.path.join(param['dataroot'], 'train', 'imgs'), mode='valid',
                                     margin_augmentation=False)

        logger.debug('valid dataset size: {}'.format(len(valid_dataset)))

        # Dataloader

        valid_dataloader = DataLoader(valid_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False, shuffle=False)

        logger.debug('valid loader size: {}'.format(len(valid_dataloader)))

        # set weight
        loss_fn = nn.CrossEntropyLoss().to(param['device'])
        eval_fn = accuracy_one_character


        # Local cv

        target_list = list()
        for _, targets, _ in valid_dataloader:
            targets = targets.argmax(dim=2)
            target_list.append(targets)
        target_list = torch.cat(target_list)

        mb = master_bar(range(snap_range[0], snap_range[1]))
        valid_logit_dict = dict()
        init = True
        
        for i in mb:
            print('load weight: {}'.format(os.path.join(rootdir, f'best_loss_{i+1}.pth')))
            model.load_state_dict(torch.load(os.path.join(rootdir, f'best_loss_{i+1}.pth')))
            logit_alcon_rnn(model, valid_dataloader, param['device'], valid_logit_dict, div=n_snap, init=init)
            init = False

        pred_list = torch.stack(list(valid_logit_dict.values()))
        pred_list = pred_list.softmax(dim=2)
        local_accuracy = accuracy_three_character(pred_list, target_list)
        logger.debug('LOCAL CV : {:5%}'.format(local_accuracy))
        torch.save(valid_logit_dict, os.path.join(outdir, f'fold{fold}_valid_logit.pth'))

        local_cv['fold{}'.format(fold)] = {'accuracy': local_accuracy, 'valid_size': len(valid_dataset)}


        logger.debug('=========== Prediction phrase ===========')

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
            logger.debug('load weight  :  {}'.format(os.path.join(outdir, f'best_loss_{i+1}.pth')))
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
