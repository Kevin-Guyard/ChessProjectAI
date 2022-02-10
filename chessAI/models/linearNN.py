import torch

class LinearNN(torch.nn.Module):
    
    def __init__(self, parameters):
        
        super(LinearNN, self).__init__()
        
        net = []
        
        net.append(torch.nn.Flatten())
        
        in_features = parameters['size_layer'].pop(0)
        
        for n_layer in range(0, parameters['nb_hidden_layers']):
            
            linear_layers = []
            out_features = parameters['size_layer'].pop(0)
            
            linear_layers.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            if parameters['batchnorm'].pop(0) == True:
                linear_layers.append(torch.nn.BatchNorm1d(out_features))
            linear_layers.append(torch.nn.Dropout(parameters['dropout'].pop(0)))
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