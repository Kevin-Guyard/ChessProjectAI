import torch

class LinearNN5(torch.nn.Module):
    
    def __init__(self, parameters):
        
        super(LinearNN5, self).__init__()
        
        self.flatten = torch.nn.Flatten()
        
        self.linear1 = torch.nn.Linear(in_features=parameters['input_size'], out_features=parameters['hidden_size1'])
        self.dropout1 = torch.nn.Dropout(p=parameters['dropout1'])
        if parameters['activation1'] == 'ReLU':
            self.activation1 = torch.nn.ReLU()
        elif parameters['activation1'] == 'LeakyReLU':
            self.activation1 = torch.nn.LeakyReLU(negative_slope=parameters['activation1_slope'])
        
        self.linear2 = torch.nn.Linear(in_features=parameters['hidden_size1'], out_features=parameters['hidden_size2'])
        self.dropout2 = torch.nn.Dropout(p=parameters['dropout2'])
        if parameters['activation2'] == 'ReLU':
            self.activation2 = torch.nn.ReLU()
        elif parameters['activation2'] == 'LeakyReLU':
            self.activation2 = torch.nn.LeakyReLU(negative_slope=parameters['activation2_slope'])
            
        self.linear3 = torch.nn.Linear(in_features=parameters['hidden_size2'], out_features=parameters['hidden_size3'])
        self.dropout3 = torch.nn.Dropout(p=parameters['dropout3'])
        if parameters['activation3'] == 'ReLU':
            self.activation3 = torch.nn.ReLU()
        elif parameters['activation3'] == 'LeakyReLU':
            self.activation3 = torch.nn.LeakyReLU(negative_slope=parameters['activation3_slope'])
            
        self.linear4 = torch.nn.Linear(in_features=parameters['hidden_size3'], out_features=parameters['hidden_size4'])
        self.dropout4 = torch.nn.Dropout(p=parameters['dropout4'])
        if parameters['activation4'] == 'ReLU':
            self.activation4 = torch.nn.ReLU()
        elif parameters['activation4'] == 'LeakyReLU':
            self.activation4 = torch.nn.LeakyReLU(negative_slope=parameters['activation4_slope'])
            
        self.linear5 = torch.nn.Linear(in_features=parameters['hidden_size4'], out_features=parameters['hidden_size5'])
        self.dropout5 = torch.nn.Dropout(p=parameters['dropout5'])
        if parameters['activation5'] == 'ReLU':
            self.activation5 = torch.nn.ReLU()
        elif parameters['activation5'] == 'LeakyReLU':
            self.activation5 = torch.nn.LeakyReLU(negative_slope=parameters['activation5_slope'])
            
        self.linear6 = torch.nn.Linear(in_features=parameters['hidden_size5'], out_features=1)
        self.activation6 = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.dropout3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.dropout4(x)
        x = self.activation4(x)
        x = self.linear5(x)
        x = self.dropout5(x)
        x = self.activation5(x)
        x = self.linear6(x)
        outputs = self.activation6(x)
        
        return outputs