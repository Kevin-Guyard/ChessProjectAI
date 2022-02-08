import random
from scipy.stats import loguniform

SIZE_MIN_LAYER = 1e0
SIZE_MAX_LAYER = 1e4

def get_model_parameters_LinearNN(model_name):
        
    nb_hidden_layers = int(model_name.split('LinearNN')[1])
        
    model_parameters = {
        'model_name': model_name,
        'nb_hidden_layers': nb_hidden_layers,
        'size_layer_0': 768
    }
    
    for n_layer in range(0, nb_hidden_layers):
        model_parameters['size_layer_' + str(n_layer + 1)] = int(loguniform.rvs(SIZE_MIN_LAYER, SIZE_MAX_LAYER, size=1)[0])
        model_parameters['activation_' + str(n_layer)] = 'LeakyReLU'
        model_parameters['activation_slope_' + str(n_layer)] = 0.1
        model_parameters['dropout_' + str(n_layer)] = random.random()
        model_parameters['batchnorm_' + str(n_layer)] = bool(random.randint(0, 1))
        
    return model_parameters