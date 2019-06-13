import torch
from fastprogress import progress_bar
from ..src.metrics import *

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
    print()
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