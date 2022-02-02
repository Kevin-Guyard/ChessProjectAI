from chessAI.modelHyperParameters.linear import *
from scipy.stats import loguniform
import random
import numpy as np

def get_parameters_tuning(n_method, model_name, nb_configurations=100, random_state=42):
    
    parameters_tuning = []
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    for n_configuration in range(0, nb_configurations):
        weight_decay = loguniform.rvs(1e-6, 1, size=1)[0]
        learning_rate = loguniform.rvs(1e-3, 1, size=1)[0]
        model_parameters = get_model_parameters(model_name=model_name, n_method=n_method)
        model_parameters.update({'weight_decay': weight_decay, 'learning_rate': learning_rate})
        parameters_tuning.append(model_parameters)
    
    return parameters_tuning


def get_model_parameters(model_name, n_method):
    
    if model_name == 'LogisticRegression': model_parameters = get_list_model_parameters_LogisticRegression(n_method)
    elif model_name == 'LinearNN1': model_parameters = get_list_model_parameters_LinearNN1(n_method)
    elif model_name == 'LinearNN2': model_parameters = get_list_model_parameters_LinearNN2(n_method)
    elif model_name == 'LinearNN3': model_parameters = get_list_model_parameters_LinearNN3(n_method)
    elif model_name == 'LinearNN4': model_parameters = get_list_model_parameters_LinearNN4(n_method)
    elif model_name == 'LinearNN5': model_parameters = get_list_model_parameters_LinearNN5(n_method)
        
    return model_parameters