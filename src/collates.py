import torch

def my_collate_fn(batch):
    # [images, targets] = dataset[batch_idx]
    # images : torch.tensor(3, 3, H, W)
    # targets: torch.tensor(3, num_class)
    images = []
    targets = []
    for sample in batch:
        image, target = sample
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return [images, targets]