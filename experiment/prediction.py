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
    rootdirs = post_process.get_root_dir(EXP_NO)
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


    param['batch size'] = max(param['batch size'], param['batch size'] * param['GPU'])
    test_dataset = AlconDataset(df=get_test_df(param['tabledir']),
                                augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                datadir=os.path.join(param['dataroot'], 'test', 'imgs'), mode='test')
    logger.debug('test dataset size: {}'.format(len(test_dataset)))

    test_dataloader = DataLoader(test_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                 pin_memory=False, drop_last=False, shuffle=False)
    logger.debug('test loader size: {}'.format(len(test_dataloader)))



    for fold in range(5):
        logger.debug(f'=========  FOLD : {fold}  =========')
        rootdir = rootdirs[fold]

        # Set Loader
        valid_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid == @fold'),
                                     augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                     datadir=os.path.join(param['dataroot'], 'train', 'imgs'), mode='valid',
                                     margin_augmentation=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False, shuffle=False)
        logger.debug('valid dataset size: {}'.format(len(valid_dataset)))
        logger.debug('valid loader size: {}'.format(len(valid_dataloader)))

        target_list = list()
        for _, targets, _ in valid_dataloader:
            targets = targets.argmax(dim=2)
            target_list.append(targets)
        target_list = torch.cat(target_list)

        mb = master_bar(range(snap_range[0], snap_range[1]))
        n_div = len(range(snap_range[0], snap_range[1]))

        valid_logit_dict = dict()
        test_logit_dict = dict()

        init = True
        
        for i in mb:
            print('load weight: {}'.format(os.path.join(rootdir, f'best_loss_{i+1}.pth')))
            model.load_state_dict(torch.load(os.path.join(rootdir, f'best_loss_{i+1}.pth')))
            logit_alcon_rnn(model, valid_dataloader, param['device'], valid_logit_dict, div=n_div, init=init)
            logit_alcon_rnn(model, test_dataloader, param['device'], test_logit_dict, div=n_div, init=init)
            init = False


        # calcurate local score
        pred_list = torch.stack(list(valid_logit_dict.values()))
        pred_list = pred_list.softmax(dim=2)
        local_accuracy = accuracy_three_character(pred_list, target_list)
        logger.debug('LOCAL CV : {:5%}'.format(local_accuracy))

        torch.save(valid_logit_dict, os.path.join(outdir, f'fold{fold}_valid_logit.pth'))
        torch.save(test_logit_dict, os.path.join(outdir, f'fold{fold}_prediction.pth'))
        local_cv['fold{}'.format(fold)] = {'accuracy': local_accuracy, 'valid_size': len(valid_dataset)}

    valid_logits = dict()
    test_logits = dict()

    for fold in range(5):
        path = os.path.join(outdir, f'fold{fold}_valid_logit.pth')
        logits = torch.load(path)
        for k, v in logits.items():
            valid_logits[k] = v
    valid_logits = sorted(valid_logits.items())
    valid_logits = dict((k, v) for k, v in valid_logits)
    torch.save(valid_logits, os.path.join(outdir, f'valid_logits.pth'))


    for fold in range(5):
        path = os.path.join(outdir, f'fold{fold}_prediction.pth')
        logits = torch.load(path)
        if fold == 0:
            for k, v in logits.items():
                test_logits[k] = v / 5
        else:
            for k, v in logits.items():
                test_logits[k] += v / 5

    torch.save(test_logits, os.path.join(outdir, 'test_logits.pth'))
    post_process.submission_to_df(post_process.make_submission(test_logits)).to_csv(os.path.join(outdir, 'test_prediction.csv'))

    print('success!')


if __name__ =='__main__':
    main()
