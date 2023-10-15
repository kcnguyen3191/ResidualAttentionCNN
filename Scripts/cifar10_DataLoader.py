import numpy as np
import torch, pdb
from torch.utils.data import Dataset
from Modules.Utils.Reproducibility import *

class DataLoader(Dataset):
    
    def __init__(self, indices, path, train=True):
        
        self.indices = indices
        self.path = path
        self.train = train
    def __len__(self):
        
        return len(self.indices)
    
    def load_sample(self, index):
        
        # load sample
        if self.train == True:
            use_folder = 'cifar10_train/'
        else:
            use_folder = 'cifar10_test/'

        file_path = self.path +'/'+ use_folder + "image_" + str(index) + ".npy"
        sample = np.load(file_path, allow_pickle=True).item()

        # extract data and normalize
        image = sample["images"][0]
        label = sample['labels']

        target = np.zeros((10,1))
        target[label] = 1
        target = target[:,0]
        return image, target
    
    def __getitem__(self, index, return_info=False):
        
        # load sample
        idx = self.indices[index]
        image, target = self.load_sample(idx)
        
        # convert to torch
        image = torch.tensor(image, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)

        return image, target

    
    
    
    
