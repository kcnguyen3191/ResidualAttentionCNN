import torch, pdb
import torch.nn as nn

from Modules.Layers.ResidualBlock import ResidualBlock

class BuildResidualEncoder(nn.Module):
    
    '''
    Builds a convolutional neural network (CNN) with residual units
    that compresses input images into a latent vector.
    
    Args:
        input_shape:    integer list/tuple with shape of inputs
        latent_dim:     integer dimensionality of latent space
        layers:         list of integer layer sizes
        num_res_blocks: integer number of residual blocks
        activation:     instantiated activation function
        use_batchnorm:  boolean for batchnorm usage
    
    Inputs:
        batch: torch float tensor of input images
    
    Returns:
        batch: torch float tensor of output images
    '''
    
    def __init__(self, 
                 input_shape,
                 latent_dim,
                 layers, 
                 num_res_blocks=2,
                 activation=None, 
                 use_batchnorm=True,
                 use_preactivations=True):
        
        super().__init__()
        self.input_shape = input_shape # (channels, height, width)
        self.input_channels = input_shape[0]
        self.latent_dim = latent_dim
        self.layers = layers
        self.num_res_blocks = num_res_blocks
        self.activation = activation if activation is not None else nn.ReLU()
        self.use_batchnorm = use_batchnorm
        self.use_preactivations = use_preactivations
        
        # list of operations in the model
        operations = []
        
        # first pre-process image with conv layers
        operations.append(nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.layers[0],
            kernel_size=3,
            stride=1,
            padding=1))
        if self.use_batchnorm:
            operations.append(nn.BatchNorm2d(num_features=self.layers[0]))
        operations.append(self.activation)
        operations.append(nn.Conv2d(
            in_channels=self.layers[0],
            out_channels=self.layers[0],
            kernel_size=3,
            stride=1,
            padding=1))
        if self.use_batchnorm:
            operations.append(nn.BatchNorm2d(num_features=self.layers[0]))
        operations.append(self.activation)
        self.input_channels = self.layers[0]
        
        # loop over blocks
        for i, layer in enumerate(layers): 
            
            # downsample
            operations.append(nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=layer,
                kernel_size=2,
                stride=2,
                padding=0))
            
            # apply residual blocks
            for j in range(self.num_res_blocks):
                
                # add res block 
                operations.append(ResidualBlock(
                    features=layer, 
                    activation=self.activation,
                    use_batchnorm=self.use_batchnorm,
                    use_preactivation=self.use_preactivations))
                
            # update book keeping
            self.input_channels = layer
            
        # output linear layer for latent space
        final_shape = int(self.input_shape[1] / 2**len(self.layers))
        self.output_linear = nn.Linear(
            in_features=layer * final_shape**2,
            out_features=self.latent_dim)
        #operations.append(nn.Conv2d(
        #    in_channels=self.input_channels,
        #    out_channels=self.latent_dim,
        #    kernel_size=int(self.input_shape[1] / 2**len(self.layers)),
        #    stride=1,
        #    padding=0))
                    
        # convert list to sequential model
        self.model = nn.Sequential(*operations)
        
    def forward(self, batch):
        
        # run the model and squeeze
        batch = self.output_linear(self.model(batch).view(len(batch), -1))
        
        return batch
        
        