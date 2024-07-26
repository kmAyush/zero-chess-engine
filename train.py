#!/usr/bin/env python
from torch.utils.data import Dataset
import numpy as np

class ValueDataset(Dataset):
    def __init__(self):
        data = np.load("processed/dataset_1k.npz")
        self.X = data['arr_0']
        self.Y = data['arr_1']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return {'X': self.X[index], 'Y': self.Y[index]}

data = ValueDataset()