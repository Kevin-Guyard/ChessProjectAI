from chessAI.modelHyperParameters.linearNN import get_model_parameters_LinearNN
from chessAI.modelHyperParameters.cNN import get_model_parameters_CNN
from scipy.stats import loguniform
import random
import numpy as np

def get_parameters_tuning(model_name, nb_config=100, random_state=42):
    
    parameters_tuning = []
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    for n_config in range(0, nb_config):
        weight_decay = loguniform.rvs(1e-6, 1, size=1)[0]
        learning_rate = loguniform.rvs(1e-3, 1, size=1)[0]
        model_parameters = get_model_parameters(model_name=model_name)
        model_parameters.update({'weight_decay': weight_decay, 'learning_rate': learning_rate})
        parameters_tuning.append(model_parameters)
    
    return parameters_tuning


def get_model_parameters(model_name):
    
    if 'LinearNN' in model_name:
        model_parameters = get_model_parameters_LinearNN(model_name=model_name)
    if 'CNN' in model_name:
        model_parameters = get_model_parameters_CNN(model_name=model_name)
        
    return model_parameters