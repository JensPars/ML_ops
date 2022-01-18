"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import os
import os.path
path = "/Users/jensparslov/Documents/DTU/ML_ops/dtu_mlops/s7_scalable_applications/exercise_files/lfw-deepfunneled"

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.transform = transform
        self.path = path_to_folder
        self.names = os.listdir(path_to_folder)
        self.imgs = {}
        self.ids = []
        for name in self.names:
            ls = []
            for file in os.listdir(path_to_folder + '/' + name):
                ls.append(file)
                self.ids.append(file)
            self.imgs[name] = ls
        
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.path + '/' + self.ids[index][:-9] + '/' + self.ids[index])
        return self.transform(img)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default=path, type=str)
    parser.add_argument('-num_workers', default=4, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', default=True,action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(path, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=args.num_workers)
    
    #if args.visualize_batch:
    #    # TODO: visualize a batch of images
    #    pass
        
    if args.get_timing:
        # lets do so repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')
