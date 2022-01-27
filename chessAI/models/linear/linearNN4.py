import torch

class LinearNN4(torch.nn.Module):
    
    def __init__(self, parameters):
        
        super(LinearNN4, self).__init__()
        
        self.linear1 = torch.nn.Linear(in_features=parameters['input_size'], out_features=parameters['hidden_size1'])
        if parameters['activation1'] == 'ReLU':
            self.activation1 = torch.nn.ReLU()
        elif parameters['activation1'] == 'LeakyReLU':
            self.activation1 = torch.nn.LeakyReLU(negative_slope=parameters['activation1_slope'])
        
        self.linear2 = torch.nn.Linear(in_features=parameters['hidden_size1'], out_features=parameters['hidden_size2'])
        if parameters['activation2'] == 'ReLU':
            self.activation2 = torch.nn.ReLU()
        elif parameters['activation2'] == 'LeakyReLU':
            self.activation2 = torch.nn.LeakyReLU(negative_slope=parameters['activation2_slope'])
            
        self.linear3 = torch.nn.Linear(in_features=parameters['hidden_size2'], out_features=parameters['hidden_size3'])
        if parameters['activation3'] == 'ReLU':
            self.activation3 = torch.nn.ReLU()
        elif parameters['activation3'] == 'LeakyReLU':
            self.activation3 = torch.nn.LeakyReLU(negative_slope=parameters['activation3_slope'])
            
        self.linear4 = torch.nn.Linear(in_features=parameters['hidden_size3'], out_features=parameters['hidden_size4'])
        if parameters['activation4'] == 'ReLU':
            self.activation4 = torch.nn.ReLU()
        elif parameters['activation4'] == 'LeakyReLU':
            self.activation4 = torch.nn.LeakyReLU(negative_slope=parameters['activation4_slope'])
            
        self.linear5 = torch.nn.Linear(in_features=parameters['hidden_size4'], out_features=1)
        self.activation5 = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        
        x1 = self.activation1(self.linear1(x))
        x2 = self.activation2(self.linear2(x1))
        x3 = self.activation3(self.linear3(x2))
        x4 = self.activation4(self.linear4(x3))
        outputs = self.activation5(self.linear5(x4))
        
        return outputs