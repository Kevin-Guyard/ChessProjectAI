import torch

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, parameters):
        
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=parameters['input_size'], out_features=1)
        self.activation = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        
        outputs = self.activation(self.linear(x))
        
        return outputs