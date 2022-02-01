import torch

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, parameters):
        
        super(LogisticRegression, self).__init__()
        
        self.flatten = torch.nn.Flatten()
        
        self.linear1 = torch.nn.Linear(in_features=parameters['input_size'], out_features=1)
        self.activation1 = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        
        x = self.flatten(x)
        x = self.linear1(x)
        outputs = self.activation1(x)
        
        return outputs