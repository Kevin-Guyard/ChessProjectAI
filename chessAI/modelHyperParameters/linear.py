import random
from scipy.stats import loguniform

def get_model_parameters_LinearNN(model_name, n_method):
    
    if n_method == 1: 
        input_size = 8 * 8 * 12
    elif n_method == 2: 
        input_size = 8 * 8 * 6
    elif n_method == 3: 
        input_size = 8 * 8 * 4
    elif n_method == 4: 
        input_size = 8 * 8 * 2
        
    nb_hidden_layers = int(model_name.split('LinearNN')[1])
        
    model_parameters = {
        'model_name': model_name,
        'nb_hidden_layers': nb_hidden_layers,
        'size_layer_0': input_size
    }
    
    for n_layer in range(0, nb_hidden_layers):
        model_parameters['size_layer_' + str(n_layer + 1)] = int(loguniform.rvs(1e0, 1e4, size=1)[0])
        model_parameters['activation_' + str(n_layer)] = 'LeakyReLU'
        model_parameters['activation_slope_' + str(n_layer)] = 0.1
        model_parameters['dropout_' + str(n_layer)] = random.random()
        model_parameters['batchnorm_' + str(n_layer)] = bool(random.randint(0, 1))
        
    return model_parameters