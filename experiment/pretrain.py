import os
import sys
sys.path.append('../')
import gc
import yaml
import datetime
import logging

import torch.nn as nn
from torch.utils.data import DataLoader
import tensorboardX as tbx
from fastprogress import progress_bar, master_bar

from models import *
from src.augmentation import get_test_augmentation, get_train_augmentation
from src.dataset import AlconDataset, KanaDataset
from src.metrics import *
from src.utils import *
from src.collates import *
from src.scheduler import *
from train_methods import *

# TODO: dropout
EXP_NAME = str(os.path.basename(__file__).split('.')[-2])

def main():
    now = datetime.datetime.now()
    now_date = '{}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

    # set logger
    logger = logging.getLogger("Log")
    logger.setLevel(logging.DEBUG)

    handler_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    print('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    with open('../params/{}.yaml'.format(EXP_NAME), "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    param['date'] = now_date
    # seed set
    seed_setting(param['seed'])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True



    # /mnt/hdd1/alcon2019/ + exp0/ + 2019-mm-dd_hh-mm-ss/ + foldN
    outdir = os.path.join(param['save path'], EXP_NAME , param['model'], now_date)
    if os.path.exists(param['save path']):
        os.makedirs(outdir, exist_ok=True)
    else:
        print("Not find {}".format(param['save path']))
        raise FileNotFoundError


    file_handler = logging.FileHandler(os.path.join(outdir, 'experiment.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)

    logger.debug('=============  Pre-training {}  ============='.format(param['model']))
    logger.debug('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))


    # Dataset

    param['batch size'] = max(param['batch size'], param['batch size'] * param['GPU'])
    if param['debug']:
        train_dataset = KanaDataset(df=get_char_df(param['tabledir']).query('valid != 0').iloc[:param['batch size']],
                                     augmentation=get_train_augmentation(get_resolution(param['resolution'])),
                                     datadir=os.path.join(param['dataroot']))

        valid_dataset = KanaDataset(df=get_char_df(param['tabledir']).query('valid == 0').iloc[:param['batch size']],
                                     augmentation=get_train_augmentation(get_resolution(param['resolution'])),
                                     datadir=os.path.join(param['dataroot']))
    else:
        train_dataset = KanaDataset(df=get_char_df(param['tabledir']).query('valid != 0'),
                                    augmentation=get_train_augmentation(get_resolution(param['resolution'])),
                                    datadir=os.path.join(param['dataroot']))

        valid_dataset = KanaDataset(df=get_char_df(param['tabledir']).query('valid == 0'),
                                    augmentation=get_train_augmentation(get_resolution(param['resolution'])),
                                    datadir=os.path.join(param['dataroot']))

    logger.debug('train dataset size: {}'.format(len(train_dataset)))
    logger.debug('valid dataset size: {}'.format(len(valid_dataset)))

    # Dataloader


    train_dataloader = DataLoader(train_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                  pin_memory=False, drop_last=False, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                  pin_memory=False, drop_last=False)

    logger.debug('train loader size: {}'.format(len(train_dataloader)))
    logger.debug('valid loader size: {}'.format(len(valid_dataloader)))

    # model
    model_name = param['model']
    if param['model'] == 'resnet18':
        model = resnet18(pretrained=True, num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'resnet34':
        model = resnet34(pretrained=True, num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'resnet50':
        model = resnet50(pretrained=True, num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'resnet101':
        model = resnet101(pretrained=True, num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'resnet152':
        model = resnet152(pretrained=True, num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'resnext50_32x4d':
        model = resnext50_32x4d(pretrained=True, num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'resnext101_32x8d':
        model = resnext101_32x8d(pretrained=True, num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'se_resnet50':
        model = se_resnet50(num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'se_resnet101':
        model = se_resnet101(num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'se_resnet152':
        model = se_resnet152(num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'se_resnext50_32x4d':
        model = se_resnext50_32x4d(num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'se_resnext101_32x4d':
        model = se_resnext101_32x4d(num_classes=48, dropout=param['dropout'])
    elif param['model'] == 'senet':
        model = senet154(num_classes=48, dropout=param['dropout'])
    else:
        raise NotImplementedError

    param['model'] = model.__class__.__name__

    # optim
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-5, nesterov=False)


    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=len(train_dataloader)*3, T_mult=2, eta_max=0.1, T_up=500)

    model = model.to(param['device'])
    if param['GPU'] > 1:
        model = nn.DataParallel(model)

    loss_fn = nn.CrossEntropyLoss().to(param['device'])
    eval_fn = accuracy

    max_char_acc = -1.
    min_loss = 10**5


    writer = tbx.SummaryWriter("../log/pretrain/{}/{}".format(model_name, now_date))

    for key, val in param.items():
        writer.add_text('data/hyperparam/{}'.format(key), str(val), 0)


    mb = master_bar(range(3 + 3*2 + 3*4)) # 21

    for epoch in mb:
        avg_train_loss = alcon_1char_train(model, optimizer, train_dataloader, param['device'],
                                       loss_fn, eval_fn, epoch, scheduler=scheduler, writer=writer, parent=mb) #ok

        avg_valid_loss, avg_valid_accuracy = alcon_1char_valid(model, valid_dataloader, param['device'], loss_fn, eval_fn)

        writer.add_scalars("data/metric/valid", {
            'loss': avg_valid_loss,
            'accuracy': avg_valid_accuracy,
        }, epoch)

        logger.info('======================== epoch {} ========================'.format(epoch+1))
        logger.info('lr              : {:.5f}'.format(scheduler.get_lr()[0]))
        logger.info('loss            : train={:.5f}  , test={:.5f}'.format(avg_train_loss, avg_valid_loss))
        logger.info('acc             :                   test={:.3%}'.format(avg_valid_accuracy))

        if min_loss > avg_valid_loss:
            logger.debug('update best loss:  {:.5f} ---> {:.5f}'.format(min_loss, avg_valid_loss))
            min_loss = avg_valid_loss
            torch.save(model.state_dict(), os.path.join(outdir, 'best_loss.pth'))

        if max_char_acc < avg_valid_accuracy:
            logger.debug('update best acc:  {:.3%} ---> {:.3%}'.format(max_char_acc, avg_valid_accuracy))
            max_char_acc = avg_valid_accuracy
            torch.save(model.state_dict(), os.path.join(outdir, 'best_acc.pth'))

        if 0:
            if scheduler is not None:
                if writer is not None:
                    writer.add_scalar("data/learning rate", scheduler.get_lr()[0], epoch)
                scheduler.step()

    writer.add_scalars("data/metric/valid", {
        'best loss': min_loss,
        'best accuracy': max_char_acc,
    })

    logger.debug('================  FINISH  TRAIN  ================')
    logger.debug('Result')
    logger.debug('Best loss : {}'.format(min_loss))
    logger.debug('Best 1 acc : {}'.format(max_char_acc))
    writer.export_scalars_to_json(os.path.join(outdir, 'history.json'))
    writer.close()

    print('success!')


if __name__ =='__main__':
    main()