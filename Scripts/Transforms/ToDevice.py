class ToDevice(object):
    
    '''
    Assigns torch tensors in sample to torch device.
    
    Args:
        device (device): torch device (e.g. cuda:0 or cpu)
        keys (list):     optional keywords if inputting dictionary
    '''
    
    def __init__(self, device, keys=None):
        
        self.device = device
        self.keys = keys
    
    def __call__(self, sample):
        
        # convert to torch tensors 
        if self.keys is None:
            sample = sample.to(self.device)
        else:
            for key in self.keys:
                sample[key] = sample[key].to(self.device)
        
        return sample