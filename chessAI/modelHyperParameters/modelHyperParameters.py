from chessAI.modelHyperParameters.linear import *

def get_parameters_tuning(n_method):
    
    parameters_tuning = []
    models_name = get_models_name()
    weights_decay = get_weights_decay()
    learning_rates = get_learning_rates()
    
    for model_name in models_name:
        list_model_parameters = get_list_model_parameters(model_name=model_name, n_method=n_method)
        shape_X = get_shape_X(model_name, n_method)
        for model_parameters in list_model_parameters:
            for learning_rate in learning_rates:
                for weight_decay in weights_decay:
                    dic_parameters = {'learning_rate': learning_rate, 'weight_decay': weight_decay, 'shape_X': shape_X}
                    dic_parameters.update(model_parameters)
                    parameters_tuning.append(dic_parameters)
                    
    return parameters_tuning


def get_models_name():
    
    models_name = [
        'LogisticRegression',
        'LinearNN1',
        'LinearNN2',
        'LinearNN3',
        'LinearNN4',
        'LinearNN5'
    ]
    
    return models_name


def get_list_model_parameters(model_name, n_method):
    
    if model_name == 'LogisticRegression': model_parameters = get_list_model_parameters_LogisticRegression(n_method)
    elif model_name == 'LinearNN1': model_parameters = get_list_model_parameters_LinearNN1(n_method)
    elif model_name == 'LinearNN2': model_parameters = get_list_model_parameters_LinearNN2(n_method)
    elif model_name == 'LinearNN3': model_parameters = get_list_model_parameters_LinearNN3(n_method)
    elif model_name == 'LinearNN4': model_parameters = get_list_model_parameters_LinearNN4(n_method)
    elif model_name == 'LinearNN5': model_parameters = get_list_model_parameters_LinearNN5(n_method)
        
    return model_parameters


def get_shape_X(model_name, n_method):
    
    if model_name == 'LogisticRegression': shape_X = get_shape_X_linear(n_method)
    elif model_name == 'LinearNN1': shape_X = get_shape_X_linear(n_method)
    elif model_name == 'LinearNN2': shape_X = get_shape_X_linear(n_method)
    elif model_name == 'LinearNN3': shape_X = get_shape_X_linear(n_method)
    elif model_name == 'LinearNN4': shape_X = get_shape_X_linear(n_method)
    elif model_name == 'LinearNN5': shape_X = get_shape_X_linear(n_method)
        
    return shape_X


def get_weights_decay():
    
    weights_decay = [1e-3, 1e-2, 1e-1, 1e0]
    
    return weights_decay

def get_learning_rates():
    
    learning_rates = [1e-3, 1e-2, 1e-1, 1e0]
    
    return learning_rates