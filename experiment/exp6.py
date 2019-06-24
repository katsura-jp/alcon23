# For Attention Branch Network

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

from models import ABN_ResNetLSTM as ResNetLSTM
from src.augmentation import get_test_augmentation, get_train_augmentation
from src.dataset import AlconDataset, KanaDataset
from src.metrics import *
from src.utils import *
from src.collates import *
from train_methods import *

EXP_NO = os.path.basename(__file__).split('.')[0][3:]
EXP_NAME = str(os.path.basename(__file__).split('.')[-2])

def main():
    now = datetime.datetime.now()
    now_date = '{}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

    # set logger
    logger = logging.getLogger("Log")
    logger.setLevel(logging.DEBUG)

    handler_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.DEBUG)
    # stream_handler.setFormatter(handler_format)
    # logger.addHandler(stream_handler)


    print('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    with open('../params/exp{}.yaml'.format(EXP_NO), "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    param['date'] = now_date
    # seed set
    seed_setting(param['seed'])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    local_cv = dict()

    for fold in param['fold']:
        # /mnt/hdd1/alcon2019/ + exp0/ + 2019-mm-dd_hh-mm-ss/ + foldN
        outdir = os.path.join(param['save path'], EXP_NAME ,now_date, 'fold{}'.format(fold))
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


        # Dataset

        param['batch size'] = max(param['batch size'], param['batch size'] * param['GPU'])
        if param['debug']:
            train_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid != @fold').iloc[:param['batch size']],
                                         augmentation=get_train_augmentation(*get_resolution(param['resolution'])),
                                         datadir=os.path.join(param['dataroot'],'train','imgs'), mode='train')

            valid_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid == @fold').iloc[:param['batch size']],
                                         augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                         datadir=os.path.join(param['dataroot'],'train','imgs'), mode='valid')
        else:
            train_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid != @fold'),
                                         augmentation=get_train_augmentation(*get_resolution(param['resolution'])),
                                         datadir=os.path.join(param['dataroot'], 'train', 'imgs'), mode='train', margin_augmentation=False)

            valid_dataset = AlconDataset(df=get_train_df(param['tabledir']).query('valid == @fold'),
                                         augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                         datadir=os.path.join(param['dataroot'], 'train', 'imgs'), mode='valid')
        logger.debug('train dataset size: {}'.format(len(train_dataset)))
        logger.debug('valid dataset size: {}'.format(len(valid_dataset)))

        # Dataloader


        train_dataloader = DataLoader(train_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False)

        logger.debug('train loader size: {}'.format(len(train_dataloader)))
        logger.debug('valid loader size: {}'.format(len(valid_dataloader)))

        # model
        model = ResNetLSTM(num_classes=48, hidden_size=512, bidirectional=True, load_weight=param['load weight'], dropout=param['dropout'])

        param['model'] = model.__class__.__name__

        # optim
        if param['optim'].lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=param['lr'], momentum=0.9,
                                        weight_decay=1e-5, nesterov=False)
        elif param['optim'].lower() == 'adam':
            optimizer = torch.optim.SGD(model.parameters(), lr=param['lr'])
        else:
            raise NotImplementedError

        # scheduler
        scheduler = eval(param['scheduler'])


        model = model.to(param['device'])
        if param['GPU'] > 0:
            model = nn.DataParallel(model)

        loss_fn = nn.CrossEntropyLoss().to(param['device'])
        eval_fn = accuracy_one_character

        max_char_acc = -1.
        max_3char_acc = -1.
        min_loss = 10**5


        writer = tbx.SummaryWriter("../log/exp{}/{}/fold{}".format(EXP_NO, now_date, fold))

        for key, val in param.items():
            writer.add_text('data/hyperparam/{}'.format(key), str(val), 0)


        mb = master_bar(range(param['epoch']))
        for epoch in mb:
            avg_train_loss, avg_train_accuracy, avg_three_train_acc = train_alcon_rnn(model, optimizer, train_dataloader, param['device'],
                                           loss_fn, eval_fn, epoch, scheduler=None, writer=writer, parent=mb) #ok

            avg_valid_loss, avg_valid_accuracy, avg_three_valid_acc = valid_alcon_rnn(model, valid_dataloader, param['device'],
                                                                                  loss_fn, eval_fn)

            writer.add_scalars("data/metric/valid", {
                'loss': avg_valid_loss,
                'accuracy': avg_valid_accuracy,
                '3accuracy': avg_three_valid_acc
            }, epoch)

            logger.debug('======================== epoch {} ========================'.format(epoch+1))
            logger.debug('lr              : {:.5f}'.format(scheduler.get_lr()[0]))
            logger.debug('loss            : train={:.5f}  , test={:.5f}'.format(avg_train_loss, avg_valid_loss))
            logger.debug('acc(per 1 char) : train={:.3%}  , test={:.3%}'.format(avg_train_accuracy, avg_valid_accuracy))
            logger.debug('acc(per 3 char) : train={:.3%}  , test={:.3%}'.format(avg_three_train_acc, avg_three_valid_acc))

            if min_loss > avg_valid_loss:
                logger.debug('update best loss:  {:.5f} ---> {:.5f}'.format(min_loss, avg_valid_loss))
                min_loss = avg_valid_loss
                torch.save(model.state_dict(), os.path.join(outdir, 'best_loss.pth'))

            if max_char_acc < avg_valid_accuracy:
                logger.debug('update best acc per 1 char:  {:.3%} ---> {:.3%}'.format(max_char_acc, avg_valid_accuracy))
                max_char_acc = avg_valid_accuracy
                torch.save(model.state_dict(), os.path.join(outdir, 'best_acc.pth'))

            if max_3char_acc < avg_three_valid_acc:
                logger.debug('update best acc per 3 char:  {:.3%} ---> {:.3%}'.format(max_3char_acc , avg_three_valid_acc))
                max_3char_acc = avg_three_valid_acc
                torch.save(model.state_dict(), os.path.join(outdir, 'best_3acc.pth'))

            if 1:
                if scheduler is not None:
                    if writer is not None:
                        writer.add_scalar("data/learning rate", scheduler.get_lr()[0], epoch)
                    scheduler.step()

        writer.add_scalars("data/metric/valid", {
            'best loss': min_loss,
            'best accuracy': max_char_acc,
            'best 3accuracy': max_3char_acc
        })

        logger.debug('================  FINISH  TRAIN  ================')
        logger.debug('Result')
        logger.debug('Best loss : {}'.format(min_loss))
        logger.debug('Best 1 acc : {}'.format(max_char_acc))
        logger.debug('Best 3 acc : {}'.format(max_3char_acc))
        writer.export_scalars_to_json(os.path.join(outdir, 'history.json'))
        writer.close()


        local_cv['fold{}'.format(fold)] = {'accuracy' : max_3char_acc, 'valid_size' : len(valid_dataset)}


        del train_dataset, valid_dataset
        del train_dataloader, valid_dataloader
        del scheduler, optimizer
        gc.collect()


        logger.debug('=========== Prediction phrase ===========')
        logger.debug('load weight  :  {}'.format(os.path.join(outdir, 'best_3acc.pth')))
        model.load_state_dict(torch.load(os.path.join(outdir, 'best_3acc.pth')))

        if param['debug']:
            test_dataset = AlconDataset(df=get_test_df(param['tabledir']).iloc[:param['batch size']],
                                        augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                        datadir=os.path.join(param['dataroot'], 'test', 'imgs'), mode='test')
        else:
            test_dataset = AlconDataset(df=get_test_df(param['tabledir']),
                                        augmentation=get_test_augmentation(*get_resolution(param['resolution'])),
                                        datadir=os.path.join(param['dataroot'], 'test', 'imgs'), mode='test')


        test_dataloader = DataLoader(test_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                     pin_memory=False, drop_last=False)
        logger.debug('test dataset size: {}'.format(len(test_dataset)))
        logger.debug('test loader size: {}'.format(len(test_dataloader)))

        output_list = pred_alcon_rnn(model, test_dataloader, param['device'])
        torch.save(output_list, os.path.join(outdir, 'prediction.pth'))
        pd.DataFrame(output_list).drop('logit', axis=1).sort_values('ID').set_index('ID').to_csv(os.path.join(outdir, 'test_prediction.csv'))
        logger.debug('success!')
        logger.removeHandler(file_handler)

        del test_dataset, test_dataloader
        gc.collect()


    # Ensemble
    print('======== Ensemble phase =========')
    prediction_dict = dict()
    mb = master_bar(param['fold'])

    print('======== Load Vector =========')
    for i, fold in enumerate(mb):
        outdir = os.path.join(param['save path'], EXP_NAME, now_date,'fold{}'.format(fold))
        prediction = torch.load(os.path.join(outdir, 'prediction.pth'))
        # prediction is list
        # prediction[0] = {'ID' : 0, 'logit' torch.tensor, ...}
        if i == 0:
            for preds in progress_bar(prediction, parent=mb):
                prediction_dict[preds['ID']] = preds['logit'] / len(param['fold'])
        else:
            for preds in progress_bar(prediction, parent=mb):
                prediction_dict[preds['ID']] += preds['logit'] / len(param['fold'])

    outdir = os.path.join(param['save path'], EXP_NAME, now_date)

    file_handler = logging.FileHandler(os.path.join(outdir, 'result.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(file_handler)
    logger.info(' ==========  RESULT  ========== \n')

    cv = 0.0
    train_data_size = 0
    for fold in param['fold']:
        acc = local_cv['fold{}'.format(fold)]['accuracy']
        valid_size = local_cv['fold{}'.format(fold)]['valid_size']
        train_data_size += valid_size
        logger.info(' fold {} :  {:.3%} \n'.format(fold, acc))
        cv += acc * valid_size
    logger.info(' Local CV : {:.3%} \n'.format(cv / train_data_size))
    logger.info(' ============================== \n')

    logger.removeHandler(file_handler)


    torch.save(prediction_dict, os.path.join(outdir, 'prediction.pth'))

    print('======== make submittion file =========')
    vocab = get_vocab(param['vocabdir'])
    submit_list = list()
    for ID, logits in progress_bar(prediction_dict.items()):
        submit_dict = dict()
        submit_dict["ID"] = ID
        preds = logits.softmax(dim=1).argmax(dim=1)
        submit_dict["Unicode1"] = vocab['index2uni'][preds[0]]
        submit_dict["Unicode2"] = vocab['index2uni'][preds[1]]
        submit_dict["Unicode3"] = vocab['index2uni'][preds[2]]
        submit_list.append(submit_dict)
    print()

    pd.DataFrame(submit_list).sort_values('ID').set_index('ID').to_csv(os.path.join(outdir, 'test_prediction.csv'))

    import zipfile
    with zipfile.ZipFile(os.path.join(outdir,'submit_{}_{}.zip'.format(EXP_NAME, now_date)), 'w') as zf:
        zf.write(os.path.join(outdir, 'test_prediction.csv'))

    print('success!')


if __name__ =='__main__':
    main()
