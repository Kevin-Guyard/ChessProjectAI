import torch

class CNN(torch.nn.Module):
    
    def __init__(self, parameters):
        
        super(CNN, self).__init__()
        
        net = []
        
        in_channels = parameters['channels'].pop(0)
        
        for n_conv_layer in range(0, parameters['nb_conv_layers']):
            
            conv_layers = []
            out_channels = parameters['channels'].pop(0)
            
            conv_layers.append(torch.nn.Conv2d(in_channels=in_channels, \
                                               out_channels=out_channels, \
                                               kernel_size=parameters['kernel_conv'].pop(0), \
                                               stride=parameters['stride_conv'].pop(0), \
                                               padding=parameters['padding_conv'].pop(0)))
            if parameters['is_conv_bis'].pop(0) == True:
                in_channels = out_channels
                out_channels = parameters['channels'].pop(0)
                conv_layers.append(torch.nn.Conv2d(in_channels=in_channels, \
                                                   out_channels=out_channels, \
                                                   kernel_size=parameters['kernel_conv'].pop(0), \
                                                   stride=parameters['stride_conv'].pop(0), \
                                                   padding=parameters['padding_conv'].pop(0)))
            if parameters['batchnorm_conv'].pop(0) == True:
                conv_layers.append(torch.nn.BatchNorm2d(num_features=out_channels))
            conv_layers.append(torch.nn.Dropout(parameters['dropout_conv'].pop(0)))
            conv_layers.append(torch.nn.ReLU())
            if parameters['is_pool_max'].pop(0) == True:
                conv_layers.append(torch.nn.MaxPool2d(kernel_size=parameters['kernel_pool'].pop(0), \
                                                      stride=parameters['stride_pool'].pop(0), \
                                                      padding=parameters['padding_pool'].pop(0)))
            else:
                conv_layers.append(torch.nn.AvgPool2d(kernel_size=parameters['kernel_pool'].pop(0), \
                                                      stride=parameters['stride_pool'].pop(0), \
                                                      padding=parameters['padding_pool'].pop(0)))
            
            conv = torch.nn.Sequential(*conv_layers)
            net.append(conv)
            
            in_channels = out_channels
            
        net.append(torch.nn.Flatten())
        
        in_features = parameters['size_linear_layer'].pop(0)
        
        for n_linear_layer in range(0, parameters['nb_linear_layers']):
            
            linear_layers = []
            out_features = parameters['size_linear_layer'].pop(0)
            
            linear_layers.append(torch.nn.Linear(in_features=in_features, \
                                                 out_features=out_features))
            if parameters['batchnorm_linear'].pop(0) == True:
                linear_layers.append(torch.nn.BatchNorm1d(out_features))
            linear_layers.append(torch.nn.Dropout(parameters['dropout_linear'].pop(0)))
            linear_layers.append(torch.nn.ReLU())
                
            linear = torch.nn.Sequential(*linear_layers)
            net.append(linear)
            
            in_features = out_features
            
        net.append(torch.nn.Linear(in_features=in_features, out_features=1))
        net.append(torch.nn.Sigmoid())
        
        self._net = torch.nn.Sequential(*net)
        
        
    def forward(self, x):
        
        outputs = self._net(x)
        
        return outputs