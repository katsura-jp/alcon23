import os
import PIL.Image as Image
import numpy as np
import torch
from torch.utils.data import Dataset


class StackingDataset(Dataset):
    def __init__(self, df, logit_path, mode='train', onehot=True):
        super(StackingDataset, self).__init__()
        self.df = df
        self.mode = mode
        self.onehot = onehot
        self.logits = torch.load(logit_path)
        self.num_classes = 48

    def __getitem__(self, index):
        _index = self.df.index[index]
        inputs = self.logits[_index]

        if self.mode == 'train':
            targets = []
            for i in range(1, 4):
                label = int(self.df.loc[_index, f"target{i}"])
                if self.onehot:
                    target = np.zeros(self.num_classes, dtype=np.float32)
                    target[label] = 1.0
                else:
                    target = label
                targets.append(target)
            targets = np.array(targets)
            targets = torch.from_numpy(targets)


        elif self.mode == 'valid':
            targets = []
            for i in range(1, 4):
                label = int(self.df.loc[_index, f"target{i}"])
                if self.onehot:
                    target = np.zeros(self.num_classes, dtype=np.float32)
                    target[label] = 1.0
                else:
                    target = label
                targets.append(target)
            targets = np.array(targets)
            targets = torch.from_numpy(targets)

        elif self.mode == 'test':
            targets = 0

        return inputs, targets, _index
    def __len__(self):
        return len(self.df)