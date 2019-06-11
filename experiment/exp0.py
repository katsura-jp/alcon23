import os
import sys
sys.append('../')
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

from models import *
from src.augmentation import get_test_augmentation, get_train_augmentation
from src.dataset import AlconDataset, KanaDataset
from src.metrics import *
from src.utils import *

def main():
    print(datetime.datetime.now())
    with open('../params/exp0.yaml', "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)

    # seed set
    seed_setting(param['seed'])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True



    # Dataset
    train_dataset = AlconDataset(df=get_train_df().query('valid != @param["fold"]'),
                                 augmentation=get_train_augmentation(), datadir=param['datadir'], mode='train')
    valid_dataset = AlconDataset(df=get_train_df().query('valid == @param["fold"]'),
                                 augmentation=get_test_augmentation(), datadir=param['datadir'], mode='valid')

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                  pin_memory=False, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=param['batch size'], num_workers=param['thread'],
                                  pin_memory=False, drop_last=False)

    # model
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 48)

    param['model'] = str(model)

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
    writer.add_scalars('hyperparam',param, 0)

    model = model.to(param['device'])
    loss_fn = torch.nn.CrossEntropyLoss().to(param['device'])
    eval_fn = accuracy

    for epoch in range(param['epoch']):
        avg_train_loss, avg_train_accuracy = train(model, optimizer, train_dataloader, param['device'],
                                       loss_fn, eval_fn, epoch, scheduler, writer)

        avg_valid_loss, avg_valid_accuracy = valid(model, valid_dataloader, param['device'], loss_fn, eval_fn)

        writer.add_scalars("data/metric", {
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy
        }, epoch)


    writer.close()


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
                writer.add_scalar("data/lr", scheduler.get_lr()[0], step + epoch*len(dataloader))
            scheduler.step()

        writer.add_scalars("data/metric", {
            'train_loss': loss.item(),
            'train_accuracy': acc
        }, step + epoch*len(dataloader))

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy

def train_alcon(model, optimizer, dataloader, device, loss_fn, eval_fn, epoch, scheduler=None, writer=None):
    model.train()
    avg_loss = 0
    avg_accuracy = 0
    three_char_accuracy = 0
    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs
        targets = targets
        _avg_loss = 0
        _avg_accuracy = 0
        pred = torch.zeros(targets.size())
        for i in range(3):
            _inputs = inputs[:, i].to(device)
            _targets = inputs[:, i].to(device)
            optimizer.zero_grad()
            logits = model(_inputs)
            _preds = logits.softmax(dim=1)
            pred[:, i] = _preds
            loss = loss_fn(logits, _targets.argmax(dim=1))
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _avg_loss += loss.item()
            acc = eval_fn(_preds, targets)
            _avg_accuracy += acc
        _avg_loss /= 3
        avg_loss += _avg_loss
        _avg_accuracy /= 3
        avg_accuracy += _avg_accuracy

        #TODO:ここに三文字の精度
        three_char_accuracy = accuracy_three_character(pred, targets, mean=True)
        if scheduler is not None:
            if writer is not None:
                writer.add_scalar("data/lr", scheduler.get_lr()[0], step + epoch*len(dataloader))
            scheduler.step()

        writer.add_scalars("data/metric", {
            'train_loss': loss.item(),
            'train_accuracy': acc,
            'train_3char_accuracy': three_char_accuracy
        }, step + epoch*len(dataloader))

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy


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


def valid_alcon(model, dataloader, device, loss_fn, eval_fn):
    model.eval()
    avg_loss = 0
    avg_accuracy = 0
    avg_3char_acc = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs
            targets = targets
            _avg_loss = 0
            _avg_accuracy = 0
            preds = torch.zeros(targets.size(), dtype='float32').to(device)
            for i in range(3):
                _inputs = inputs[:, i].to(device)
                _targets = targets[:, i].to(device)
                logits = model(inputs)
                _preds = logits.softmax(dim=1)
                preds[:, i] = _preds
                loss = loss_fn(logits, _targets.argmax(dim=1))
                _avg_loss += loss.item()
                _avg_accuracy += eval_fn(_preds, _targets)
            _avg_loss /= 3
            _avg_accuracy /= 3
            avg_loss += _avg_loss
            avg_accuracy += _avg_accuracy
            avg_3char_acc += accuracy_three_character(preds, targets, mean=True)
        avg_loss /= len(dataloader)
        avg_accuracy /= len(dataloader)
        avg_3char_acc /= len(dataloader)
    return avg_loss, avg_accuracy, avg_3char_acc

def seed_setting(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True