import torch

class LinearNN(torch.nn.Module):
    
    def __init__(self, parameters):
        
        super(LinearNN, self).__init__()
        
        layers = []
        
        layers.append(torch.nn.Flatten())
        
        for n_layer in range(0, parameters['nb_hidden_layers']):
            layers.append(torch.nn.Linear(in_features=parameters['size_layer_' + str(n_layer)], out_features=parameters['size_layer_' + str(n_layer + 1)]))
            if parameters['activation_' + str(n_layer)] == 'ReLU':
                layers.append(torch.nn.ReLU())
            elif parameters['activation_' + str(n_layer)] == 'LeakyReLU':
                layers.append(torch.nn.LeakyReLU(negative_slope=parameters['activation_slope_' + str(n_layer)]))
            layers.append(torch.nn.Dropout(parameters['dropout_' + str(n_layer)]))
            if parameters['batch_norm_' + str(n_layer)] == True:
                layers.append(torch.nn.BatchNorm1d(parameters['size_layer_' + str(n_layer + 1)]))
                
        layers.append(torch.nn.Linear(in_features=parameters['size_layer_' + str(parameters['nb_hidden_layers'])], out_features=1))
        layers.append(torch.nn.Sigmoid())
        
        self._net = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        
        outputs = self._net(x)
        
        return outputs