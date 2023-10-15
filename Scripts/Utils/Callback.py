import pdb

class Callback():
   
    '''
    Wrapper around custom callback functions with specifications for when to 
    use in ModelWrapper training loop.
   
    Args:
        callback   (callable): Callback function that inputs ModelWrapper.
        on_epoch_begin (bool): When to use callback. 
        on_epoch_end   (bool): When to use callback. 
        on_batch_begin (bool): When to use callback. 
        on_batch_end   (bool): When to use callback. 
        on_train_begin (bool): When to use callback. 
        on_train_end   (bool): When to use callback. 
    '''
   
    def __init__(self,
                 callback=None,
                 on_epoch_begin=False,
                 on_epoch_end=False,
                 on_batch_begin=False,
                 on_batch_end=False,
                 on_train_begin=False,
                 on_train_end=False):
        
        self.callback = callback if callback is not None else self.do_nothing
        self.on_epoch_begin = on_epoch_begin
        self.on_epoch_end = on_epoch_end
        self.on_batch_begin = on_batch_begin
        self.on_batch_end = on_batch_end
        self.on_train_begin = on_train_begin
        self.on_train_end = on_train_end
        
    def do_nothing(self, *args, **kwargs):
        pass
    
    def __call__(self, *args):
        self.callback(*args)