from chessAI.models.linearNN import LinearNN

def get_model(parameters):
        
    if 'LinearNN' in parameters['model_name'] :
        model = LinearNN(parameters)
        
    return model