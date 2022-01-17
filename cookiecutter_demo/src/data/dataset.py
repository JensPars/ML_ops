import torch
import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.input = np.load("/Users/jensparslov/Documents/DTU/ML_ops/cookiecutter_demo/data/processed/imgs.npy")
        self.labels = np.load("/Users/jensparslov/Documents/DTU/ML_ops/cookiecutter_demo/data/processed/classes.npy")

    def classes(self):
        return np.unique(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input[idx], self.labels[idx]
