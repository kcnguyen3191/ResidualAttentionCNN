import torch

class ToTensor(object):
    
    '''
    Convert numpy arrays in sample to torch tensors.
    
    Args:
        keys (list): optional keywords if inputting dictionary
    '''
    
    def __init__(self, keys=None):
        
        self.keys = keys
    
    def __call__(self, sample):
        
        # convert to torch tensors 
        if self.keys is None:
            sample = torch.tensor(sample, dtype=torch.float)
        else:
            for key in self.keys:
                sample[key] = torch.tensor(sample[key], dtype=torch.float)
        
        return sample