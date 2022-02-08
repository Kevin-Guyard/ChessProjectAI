import torch

class CNN(torch.nn.Module):
    
    def __init__(self, parameters=None):
        
        super(CNN, self).__init__()
        
        layers = []
        
        for n_conv_layer in range(0, parameters['nb_conv_layers']):
            layers.append(torch.nn.Conv2d(in_channels=parameters['channels_' + str(n_conv_layer)], \
                                          out_channels=parameters['channels_' + str(n_conv_layer + 1)], \
                                          kernel_size=parameters['kernel_conv_' + str(n_conv_layer)], \
                                          stride=parameters['stride_conv_' + str(n_conv_layer)], \
                                          padding=parameters['padding_conv_' + str(n_conv_layer)]))
            layers.append(torch.nn.ReLU())
            if parameters['batchnorm_conv_' + str(n_conv_layer)] == True:
                layers.append(torch.nn.BatchNorm2d(num_features=parameters['channels_' + str(n_conv_layer + 1)]))
            if parameters['is_pool_max_' + str(n_conv_layer)] == True:
                layers.append(torch.nn.MaxPool2d(kernel_size=parameters['kernel_pool_' + str(n_conv_layer)], \
                                                 stride=parameters['stride_pool_' + str(n_conv_layer)], \
                                                 padding=parameters['padding_pool_' + str(n_conv_layer)]))
            else:
                layers.append(torch.nn.AvgPool2d(kernel_size=parameters['kernel_pool_' + str(n_conv_layer)], \
                                                 stride=parameters['stride_pool_' + str(n_conv_layer)], \
                                                 padding=parameters['padding_pool_' + str(n_conv_layer)]))
            layers.append(torch.nn.Dropout(parameters['dropout_conv_' + str(n_conv_layer)]))
        
        layers.append(torch.nn.Flatten())
        
        for n_linear_layer in range(0, parameters['nb_linear_layers']):
            layers.append(torch.nn.Linear(in_features=parameters['size_linear_layer_' + str(n_linear_layer)], out_features=parameters['size_linear_layer_' + str(n_linear_layer + 1)]))
            if parameters['activation_linear_' + str(n_linear_layer)] == 'ReLU':
                layers.append(torch.nn.ReLU())
            elif parameters['activation_linear_' + str(n_linear_layer)] == 'LeakyReLU':
                layers.append(torch.nn.LeakyReLU(negative_slope=parameters['activation_slope_linear_' + str(n_linear_layer)]))
            layers.append(torch.nn.Dropout(parameters['dropout_linear_' + str(n_linear_layer)]))
            if parameters['batchnorm_linear_' + str(n_linear_layer)] == True:
                layers.append(torch.nn.BatchNorm1d(parameters['size_linear_layer_' + str(n_linear_layer + 1)]))
        
        layers.append(torch.nn.Linear(in_features=parameters['size_linear_layer_' + str(parameters['nb_linear_layers'])], out_features=1))
        layers.append(torch.nn.Sigmoid())
        
        self._net = torch.nn.Sequential(*layers)
        
        
    def forward(self, x):
        
        outputs = self._net(x)
        
        return outputs