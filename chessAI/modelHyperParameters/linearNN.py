import random
from scipy.stats import loguniform

SIZE_MIN_LAYER = 10
SIZE_MAX_LAYER = 1e4

def get_model_parameters_LinearNN(model_name):
        
    nb_hidden_layers = int(model_name.split('LinearNN')[1])
        
    model_parameters = {
        'model_name': model_name,
        'nb_hidden_layers': nb_hidden_layers,
        'size_layer': [768],
        'dropout': [],
        'batchnorm': []
    }
    
    for n_layer in range(0, nb_hidden_layers):
        
        size_layer = int(loguniform.rvs(SIZE_MIN_LAYER, SIZE_MAX_LAYER, size=1)[0])
        dropout = random.random()
        batchnorm = bool(random.randint(0, 1))
        
        model_parameters['size_layer'].append(size_layer)
        model_parameters['dropout'].append(dropout)
        model_parameters['batchnorm'].append(batchnorm)
        
    return model_parameters