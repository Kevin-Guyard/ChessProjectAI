from chessAI.models.linearNN import LinearNN
from chessAI.models.cNN import CNN

def get_model(parameters):
        
    if 'LinearNN' in parameters['model_name'] :
        model = LinearNN(parameters)
    if 'CNN' in parameters['model_name']:
        model = CNN(parameters)
        
    return model