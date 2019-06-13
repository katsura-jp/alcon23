import torch
from fastprogress import progress_bar

def train(model, optimizer, dataloader, device, loss_fn, eval_fn, epoch, scheduler=None, writer=None, parent=None):
    model.train()
    avg_loss = 0
    avg_accuracy = 0
    for step, (inputs, targets) in enumerate(progress_bar(dataloader, parent=parent)):
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
    print()
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