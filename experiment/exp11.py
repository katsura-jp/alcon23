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

    resume_fold = -1
    if param['resume'] is not None:
        print('--- resume ---')
        info = torch.load(os.path.join(param['resume'], 'info.pth'))
        now_date = info['now_date']
        resume_fold = info['fold']

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

        logger.debug('train dataset size: {}'.format(len(train_dataset)))
        logger.debug('valid dataset size: {}'.format(len(valid_dataset)))

        # Dataloader


        train_dataloader = DataLoader(train_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                      pin_memory=False, drop_last=False, shuffle=False)

        logger.debug('train loader size: {}'.format(len(train_dataloader)))
        logger.debug('valid loader size: {}'.format(len(valid_dataloader)))

        # model
        model = SEResNeXtGRU2(num_classes=48, hidden_size=512, bidirectional=True, load_weight=None, dropout=param['dropout'])



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

        writer = tbx.SummaryWriter("../log/exp{}/{}/fold{}".format(EXP_NO, now_date, fold))

        for key, val in param.items():
            writer.add_text('data/hyperparam/{}'.format(key), str(val), 0)



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
        resume = False
        if resume_fold == fold:
            logger.debug('########################')
            logger.debug('##      RESUME        ##')
            logger.debug('########################')
            resume = True
            resume_epoch = info['epoch']
            snapshot = info['snapshot']
            min_loss = info['min_loss']
            max_char_acc = info['max_char_acc']
            max_3char_acc = info['max_char_3acc']
            snapshot_loss = info['snapshot_loss']
            snapshot_eval = info['snapshot_eval']
            snapshot_eval3 = info['snapshot_eval3']
            logger.debug(f'epoch : {resume_epoch}')
            logger.debug(f'snapshot : {snapshot}')
            logger.debug('########################')


            model.load_state_dict(torch.load(os.path.join(outdir, 'latest.pth')))


        for epoch in mb:
            if resume and epoch <= resume_epoch:
                if epoch == resume_epoch:
                    print('set scheduler state')
                    scheduler.step((epoch+1) * len(train_dataloader))
                    print(f'lr : {scheduler.get_lr()}')
                continue

            if epoch % cycle_iter == 0 and epoch >= snap_start * cycle_iter:
                if snapshot > 1:
                    snapshot_loss_list.append(snapshot_loss)
                    snapshot_eval_list.append(snapshot_eval)
                    snapshot_eval3_list.append(snapshot_eval3)
                snapshot += 1
                snapshot_loss = 10**5
                snapshot_eval = 0.0
                snapshot_eval3 = 0.0
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
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                avg_train_loss += loss.item()
                _avg_accuracy = eval_fn(preds, targets.argmax(dim=2)).item()
                avg_train_accuracy += _avg_accuracy
                _three_char_accuracy = accuracy_three_character(preds, targets.argmax(dim=2), mean=True).item()
                avg_three_train_acc += _three_char_accuracy
                writer.add_scalar("data/learning rate", scheduler.get_lr()[0], step + epoch * len(train_dataloader))
                scheduler.step()
                writer.add_scalars("data/metric/train", {
                    'loss': loss.item(),
                    'accuracy': _avg_accuracy,
                    '3accuracy': _three_char_accuracy
                }, step + epoch * len(train_dataloader))
                if step % val_iter == 0 and step != 0:
                    avg_valid_loss, avg_valid_accuracy, avg_three_valid_acc = valid_alcon_rnn(model, valid_dataloader,
                                                                                              param['device'],
                                                                                              loss_fn, eval_fn)
                    writer.add_scalars("data/metric/valid", {
                        'loss': avg_valid_loss,
                        'accuracy': avg_valid_accuracy,
                        '3accuracy': avg_three_valid_acc
                    }, epoch)

                    logger.debug('======================== epoch {} | step {} ========================'.format(epoch + 1, step+1))
                    logger.debug('lr              : {:.5f}'.format(scheduler.get_lr()[0]))
                    logger.debug('loss            : test={:.5f}'.format(avg_valid_loss))
                    logger.debug('acc(per 1 char) : test={:.3%}'.format(avg_valid_accuracy))
                    logger.debug('acc(per 3 char) : test={:.3%}'.format(avg_three_valid_acc))

                    if min_loss > avg_valid_loss:
                        logger.debug('update best loss:  {:.5f} ---> {:.5f}'.format(min_loss, avg_valid_loss))
                        min_loss = avg_valid_loss
                        torch.save(model.state_dict(), os.path.join(outdir, 'best_loss.pth'))

                    if max_char_acc < avg_valid_accuracy:
                        logger.debug(
                            'update best acc per 1 char:  {:.3%} ---> {:.3%}'.format(max_char_acc, avg_valid_accuracy))
                        max_char_acc = avg_valid_accuracy
                        torch.save(model.state_dict(), os.path.join(outdir, 'best_acc.pth'))

                    if max_3char_acc < avg_three_valid_acc:
                        logger.debug('update best acc per 3 char:  {:.3%} ---> {:.3%}'.format(max_3char_acc,
                                                                                              avg_three_valid_acc))
                        max_3char_acc = avg_three_valid_acc
                        torch.save(model.state_dict(), os.path.join(outdir, 'best_3acc.pth'))
                    if snapshot > 0:
                        if snapshot_loss > avg_valid_loss:
                            logger.debug('[snap] update best loss:  {:.5f} ---> {:.5f}'.format(snapshot_loss, avg_valid_loss))
                            snapshot_loss = avg_valid_loss
                            torch.save(model.state_dict(), os.path.join(outdir, f'best_loss_{snapshot}.pth'))

                        if snapshot_eval < avg_valid_accuracy:
                            logger.debug(
                                '[snap] update best acc per 1 char:  {:.3%} ---> {:.3%}'.format(snapshot_eval,
                                                                                         avg_valid_accuracy))
                            snapshot_eval = avg_valid_accuracy
                            torch.save(model.state_dict(), os.path.join(outdir, f'best_acc_{snapshot}.pth'))

                        if snapshot_eval3 < avg_three_valid_acc:
                            logger.debug(
                                '[snap] update best acc per 3 char:  {:.3%} ---> {:.3%}'.format(snapshot_eval3,
                                                                                         avg_three_valid_acc))
                            snapshot_eval3 = avg_three_valid_acc
                            torch.save(model.state_dict(), os.path.join(outdir, f'best_3acc_{snapshot}.pth'))


            avg_train_loss /= len(train_dataloader)
            avg_train_accuracy /= len(train_dataloader)
            avg_three_train_acc /= len(train_dataloader)



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

            if epoch == cycle_iter * snap_start:
                torch.save(model.state_dict(), os.path.join(outdir, f'model_epoch_{cycle_iter * snap_start}.pth'))

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
            if snapshot > 0:
                if snapshot_loss > avg_valid_loss:
                    logger.debug('[snap] update best loss:  {:.5f} ---> {:.5f}'.format(snapshot_loss, avg_valid_loss))
                    snapshot_loss = avg_valid_loss
                    torch.save(model.state_dict(), os.path.join(outdir, f'best_loss_{snapshot}.pth'))

                if snapshot_eval < avg_valid_accuracy:
                    logger.debug(
                        '[snap] update best acc per 1 char:  {:.3%} ---> {:.3%}'.format(snapshot_eval, avg_valid_accuracy))
                    snapshot_eval = avg_valid_accuracy
                    torch.save(model.state_dict(), os.path.join(outdir, f'best_acc_{snapshot}.pth'))

                if snapshot_eval3 < avg_three_valid_acc:
                    logger.debug(
                        '[snap] update best acc per 3 char:  {:.3%} ---> {:.3%}'.format(snapshot_eval3, avg_three_valid_acc))
                    snapshot_eval3 = avg_three_valid_acc
                    torch.save(model.state_dict(), os.path.join(outdir, f'best_3acc_{snapshot}.pth'))

            torch.save(model.state_dict(), os.path.join(outdir, 'latest.pth'))
            torch.save({
                'now_date': now_date,
                'epoch' : epoch,
                'fold' : fold,
                'snapshot': snapshot,
                'min_loss': min_loss,
                'max_char_acc': max_char_acc,
                'max_3char_acc': max_3char_acc,
                'snapshot_loss': snapshot_loss,
                'snapshot_eval': snapshot_eval,
                'snapshot_eval3' : snapshot_eval3,
            }, os.path.join(outdir, 'info.pth'))



        snapshot_loss_list.append(snapshot_loss)
        snapshot_eval_list.append(snapshot_eval)
        snapshot_eval3_list.append(snapshot_eval3)
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

        # Local cv

        target_list = list()
        for _, targets, _ in valid_dataloader:
            targets = targets.argmax(dim=2)
            target_list.append(targets)
        target_list = torch.cat(target_list)

        mb = master_bar(range(n_snap))
        valid_logit_dict = dict()
        init = True
        for i in mb:
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
