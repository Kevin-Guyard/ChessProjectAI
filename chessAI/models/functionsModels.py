from chessAI.models.linearNN import LinearNN

def get_model(parameters):
        
    if parameters['model_name'] == 'LogisticRegression' \
    or parameters['model_name'] == 'LinearNN1' \
    or parameters['model_name'] == 'LinearNN2' \
    or parameters['model_name'] == 'LinearNN3' \
    or parameters['model_name'] == 'LinearNN4' \
    or parameters['model_name'] == 'LinearNN5':
        model = LinearNN(parameters)
        
    return model