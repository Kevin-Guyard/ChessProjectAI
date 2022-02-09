from chessAI.models.linearNN import LinearNN
from chessAI.models.cNN import CNN
from copy import deepcopy

def get_model(parameters):
    
    parameters_copy = deepcopy(parameters)
        
    if 'LinearNN' in parameters['model_name'] :
        model = LinearNN(parameters_copy)
    if 'CNN' in parameters['model_name']:
        model = CNN(parameters_copy)
        
    return model