from chessAI.models.linear import LogisticRegression, LinearNN1, LinearNN2, LinearNN3, LinearNN4, LinearNN5

def get_model(parameters):
        
    if parameters['model_name'] == 'LogisticRegression':
        model = LogisticRegression(parameters)
    elif parameters['model_name'] == 'LinearNN1':
        model = LinearNN1(parameters)
    elif parameters['model_name'] == 'LinearNN2':
        model = LinearNN2(parameters)
    elif parameters['model_name'] == 'LinearNN3':
        model = LinearNN3(parameters)
    elif parameters['model_name'] == 'LinearNN4':
        model = LinearNN4(parameters)
    elif parameters['model_name'] == 'LinearNN5':
        model = LinearNN5(parameters)
        
    return model