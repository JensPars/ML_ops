import torch
import numpy as np
import torch
import os.path as path

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        yourpath =  "data/processed/imgs.npy"
        two_up =  str(path.abspath(path.join(yourpath ,"../.."))) + "/processed/imgs.npy"
        self.input = np.load(two_up)
        yourpath =  "data/processed/classes.npy"
        two_up =  str(path.abspath(path.join(yourpath ,"../.."))) + "/processed/classes.npy"
        self.labels = np.load(two_up)

    def classes(self):
        return np.unique(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input[idx], self.labels[idx]