import torch

class LinearNN1(torch.nn.Module):
    
    def __init__(self, parameters):
        
        super(LinearNN1, self).__init__()
        
        self.flatten = torch.nn.Flatten()
        
        self.linear1 = torch.nn.Linear(in_features=parameters['input_size'], out_features=parameters['hidden_size1'])
        if parameters['activation1'] == 'ReLU':
            self.activation1 = torch.nn.ReLU()
        elif parameters['activation1'] == 'LeakyReLU':
            self.activation1 = torch.nn.LeakyReLU(negative_slope=parameters['activation1_slope'])
            
        self.linear2 = torch.nn.Linear(in_features=parameters['hidden_size1'], out_features=1)
        self.activation2 = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        outputs = self.activation2(x)
        
        return outputs