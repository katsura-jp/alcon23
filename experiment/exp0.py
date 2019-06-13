import os
import sys
sys.path.append('../')
import gc
import glob
import yaml
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import tensorboardX as tbx
from torchvision.models import resnet18
from fastprogress import progress_bar, master_bar

from models import *
from src.augmentation import get_test_augmentation, get_train_augmentation
from src.dataset import AlconDataset, KanaDataset
from src.metrics import *
from src.utils import *
from src.collates import *

#TODO:

def main():
    now = datetime.datetime.now()
    now_date = '{}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    print('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    with open('../params/exp0.yaml', "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)

    # seed set
    seed_setting(param['seed'])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    fold = param['fold']
    outdir = os.path.join(param['save path'], str(os.path.basename(__file__).split('.')[-2]) + '_fold{}'.format(fold), now_date)
    if os.path.exists(param['save path']):
        os.makedirs(outdir, exist_ok=True)
    else:
        print("Not find {}".format(param['save path']))
        raise FileNotFoundError


    # Dataset

    train_dataset = AlconDataset(df=get_train_df().query('valid != @fold'),
                                 augmentation=get_train_augmentation(),
                                 datadir=os.path.join(param['dataroot'],'train','imgs'), mode='train')
    valid_dataset = AlconDataset(df=get_train_df().query('valid == @fold'),
                                 augmentation=get_test_augmentation(),
                                 datadir=os.path.join(param['dataroot'],'train','imgs'), mode='valid')
    print('train dataset size: {}'.format(len(train_dataset)))
    print('valid dataset size: {}'.format(len(valid_dataset)))

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=param['batch size'],num_workers=param['thread'],
                                  pin_memory=False, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                  pin_memory=False, drop_last=False)

    print('train loader size: {}'.format(len(train_dataloader)))
    print('valid loader size: {}'.format(len(valid_dataloader)))

    # model
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 48)

    param['model'] = model.__class__.__name__

    # optim
    if param['optim'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=param['lr'], momentum=0.9,
                                    weight_decay=1e-5, nesterov=False)
    elif param['optim'].lower() == 'adam':
        optimizer =  torch.optim.SGD(model.parameters(), lr=param['lr'])
    else:
        raise NotImplementedError

    # scheduler
    scheduler = eval(param['scheduler'])


    writer = tbx.SummaryWriter("../log/exp0")
    for key, val in param.items():
        # print(f'{key}: {val}')
        writer.add_text('hyperparam/{}'.format(key),str(val), 0)

    model = model.to(param['device'])
    loss_fn = torch.nn.CrossEntropyLoss().to(param['device'])
    eval_fn = accuracy

    max_char_acc = 0.
    max_3char_acc = 0.
    min_loss = 10**5
    mb = master_bar(range(param['epoch']))
    for epoch in mb:
        avg_train_loss, avg_train_accuracy, avg_three_train_acc = train_alcon(model, optimizer, train_dataloader, param['device'],
                                       loss_fn, eval_fn, epoch, scheduler=None, writer=writer) #ok

        avg_valid_loss, avg_valid_accuracy, avg_three_valid_acc = valid_alcon(model, valid_dataloader, param['device'],
                                                                              loss_fn, eval_fn)

        writer.add_scalars("data/metric/valid", {
            'loss': avg_valid_loss,
            'accuracy': avg_valid_accuracy,
            '3accuracy': avg_three_valid_acc
        }, epoch)

        print('======================== epoch {} ========================'.format(epoch+1))
        print('lr              : {:.5f}'.format(scheduler.get_lr()[0]))
        print('loss            : train={:.5f}  , test={:.5f}'.format(avg_train_loss, avg_valid_loss))
        print('acc(per 1 char) : train={:.3%}  , test={:.3%}'.format(avg_train_accuracy, avg_valid_accuracy))
        print('acc(per 3 char) : train={:.3%}  , test={:.3%}'.format(avg_three_train_acc, avg_three_valid_acc))

        if min_loss > avg_valid_loss:
            print('update best loss:  {:.5f} ---> {:.5f}'.format(min_loss, avg_valid_loss))
            min_loss = avg_valid_loss
            torch.save(model.state_dict(), os.path.join(outdir, 'best_loss.pth'))

        if max_char_acc < avg_valid_accuracy:
            print('update best acc per 1 char:  {:.3%} ---> {:.3%}'.format(max_char_acc, avg_valid_accuracy))
            max_char_acc = avg_valid_accuracy
            torch.save(model.state_dict(), os.path.join(outdir, 'best_acc.pth'))

        if max_3char_acc < avg_three_valid_acc:
            print('update best acc per 3 char:  {:.3%} ---> {:.3%}'.format(max_3char_acc , avg_three_valid_acc))
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

    print('finish train')
    print('result')
    print('best loss : {}'.format(min_loss))
    print('best 1 acc : {}'.format(max_char_acc))
    print('best 3 acc : {}'.format(max_3char_acc))
    writer.export_scalars_to_json(os.path.join(outdir, 'history.json'))
    writer.close()

    #TODO: prediction, SnapShotEnsemble


def prediction():
    pass

def train(model, optimizer, dataloader, device, loss_fn, eval_fn, epoch, scheduler=None, writer=None):
    model.train()
    avg_loss = 0
    avg_accuracy = 0
    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        preds = logits.softmax(dim=1)
        loss = loss_fn(logits, targets.argmax(dim=1))
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        avg_loss += loss.item()
        acc = eval_fn(preds, targets)
        avg_accuracy += acc
        if scheduler is not None:
            if writer is not None:
                writer.add_scalar("data/learning rate", scheduler.get_lr()[0], step + epoch*len(dataloader))
            scheduler.step()

        writer.add_scalars("data/metric/train", {
            'train_loss': loss.item(),
            'train_accuracy': acc
        }, step + epoch*len(dataloader))

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy


def valid(model, dataloader, device, loss_fn, eval_fn):
    model.eval()
    avg_loss = 0
    avg_accuracy = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.softmax(dim=1)
            loss = loss_fn(logits, targets.argmax(dim=1))
            avg_loss += loss.item()
            avg_accuracy += eval_fn(preds, targets)
        avg_loss /= len(dataloader)
        avg_accuracy /= len(dataloader)

    return avg_loss, avg_accuracy


def train_alcon(model, optimizer, dataloader, device, loss_fn, eval_fn, epoch, scheduler=None, writer=None, parent=None):
    model.train()
    avg_loss = 0
    avg_accuracy = 0
    three_char_accuracy = 0
    for step, (inputs, targets) in enumerate(progress_bar(dataloader, parent=parent)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        _avg_loss = 0
        _avg_accuracy = 0
        preds = torch.zeros(targets.size()).to(device)
        for i in range(3):
            # _inputs = inputs[:, i].to(device)
            # _targets = targets[:, i].to(device)
            _inputs = inputs[:, i]
            _targets = targets[:, i]
            optimizer.zero_grad()
            logits = model(_inputs)
            _preds = logits.softmax(dim=1)
            preds[:, i] = _preds
            loss = loss_fn(logits, _targets.argmax(dim=1))
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _avg_loss += loss.item()
            acc = eval_fn(_preds, _targets.argmax(dim=1))
            _avg_accuracy += acc.item()
        _avg_loss /= 3
        avg_loss += _avg_loss
        _avg_accuracy /= 3
        avg_accuracy += _avg_accuracy

        _three_char_accuracy = accuracy_three_character(preds, targets.argmax(dim=2), mean=True).item()
        # _three_char_accuracy = accuracy_three_character(pred, targets.to(device), mean=True)

        three_char_accuracy += _three_char_accuracy
        if scheduler is not None:
            if writer is not None:
                writer.add_scalar("data/learning rate", scheduler.get_lr()[0], step + epoch*len(dataloader))
            scheduler.step()

        writer.add_scalars("data/metric/train", {
            'loss': _avg_loss,
            'accuracy': _avg_accuracy,
            '3accuracy': _three_char_accuracy
        }, step + epoch*len(dataloader))

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    three_char_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy, three_char_accuracy


def valid_alcon(model, dataloader, device, loss_fn, eval_fn):
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        avg_accuracy = 0
        three_char_accuracy = 0
        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            _avg_loss = 0
            _avg_accuracy = 0
            preds = torch.zeros(targets.size()).to(device)
            for i in range(3):
                # _inputs = inputs[:, i].to(device)
                # _targets = targets[:, i].to(device)
                _inputs = inputs[:, i]
                _targets = targets[:, i]
                logits = model(_inputs)
                _preds = logits.softmax(dim=1)
                preds[:, i] = _preds
                loss = loss_fn(logits, _targets.argmax(dim=1))
                _avg_loss += loss.item()
                acc = eval_fn(_preds, _targets.argmax(dim=1))
                _avg_accuracy += acc.item()
            _avg_loss /= 3
            avg_loss += _avg_loss
            _avg_accuracy /= 3
            avg_accuracy += _avg_accuracy

            _three_char_accuracy = accuracy_three_character(preds, targets.argmax(dim=2), mean=True).item()
            # _three_char_accuracy = accuracy_three_character(pred, targets.to(device), mean=True)

            three_char_accuracy += _three_char_accuracy

        avg_loss /= len(dataloader)
        avg_accuracy /= len(dataloader)
        three_char_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy, three_char_accuracy


def train_mixup(model, optimizer, dataloader, device, loss_fn, eval_fn, epoch, scheduler=None, writer=None):
    model.train()
    avg_loss = 0
    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2, device=device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = lam * loss_fn(logits, targets_a.argmax(dim=1))  + (1 - lam) * loss_fn(logits, targets_b.argmax(dim=1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        avg_loss += loss.item()
        if scheduler is not None:
            if writer is not None:
                writer.add_scalar("data/lr", scheduler.get_lr()[0], step + epoch*len(dataloader))
            scheduler.step()
        writer.add_scalars("data/metric/train_loss", loss.item(), step + epoch * len(dataloader))
    avg_loss /= len(dataloader)

    return avg_loss





if __name__ =='__main__':
    main()
