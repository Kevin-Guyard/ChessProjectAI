import random
from scipy.stats import loguniform

NB_MAX_CHANNELS = 48
SIZE_MAX_KERNEL_CONV = 7
STRIDE_MIN_CONV = 1
STRIDE_MAX_CONV = 3
PADDING_MIN_CONV = 0
PADDING_MAX_CONV = 3
STRIDE_MIN_POOL = 1
STRIDE_MAX_POOL = 3
SIZE_MIN_KERNEL_POOL = 1
SIZE_MAX_KERNEL_POOL = 3
PADDING_MIN_POOL = 0
PADDING_MAX_POOL = 3
SIZE_MIN_LINEAR_LAYER = 10
SIZE_MAX_LINEAR_LAYER = 1000

def get_model_parameters_CNN(model_name):
    
    nb_conv_layers = int(model_name.split('CNN')[1].split('-')[0])
    nb_linear_layers = int(model_name.split('CNN')[1].split('-')[1])
    
    model_parameters = {
        'model_name': model_name,
        'nb_conv_layers': nb_conv_layers,
        'nb_linear_layers': nb_linear_layers,
        'channels': [12],
        'kernel_conv': [],
        'stride_conv': [],
        'padding_conv': [],
        'is_conv_bis': [],
        'is_pool_max': [],
        'kernel_pool': [],
        'stride_pool': [],
        'padding_pool': [],
        'dropout_conv': [],
        'batchnorm_conv': [],
        'size_linear_layer': [],
        'dropout_linear': [],
        'batchnorm_linear': []
    }
    
    channels = 12
    kernel_conv = 1  
    dim = 8
    
    for n_conv_layer in range(0, nb_conv_layers):
        
        channels = random.randint(channels, NB_MAX_CHANNELS)
        kernel_conv = random.randint(kernel_conv, SIZE_MAX_KERNEL_CONV)
        if kernel_conv % 2 == 0:
            kernel_conv += 1
        if kernel_conv > dim:
            kernel_conv = dim
            if kernel_conv % 2 == 0:
                kernel_conv -= 1
        stride_conv = random.randint(STRIDE_MIN_CONV, STRIDE_MAX_CONV)
        padding_conv = random.randint(PADDING_MIN_CONV, min(PADDING_MAX_CONV, int(kernel_conv/2)))
    
        model_parameters['channels'].append(channels) 
        model_parameters['kernel_conv'].append(kernel_conv)
        model_parameters['stride_conv'].append(stride_conv)
        model_parameters['padding_conv'].append(padding_conv)

        dim = int((dim - kernel_conv + 2 * padding_conv) / stride_conv) + 1
        
        is_conv_bis = bool(random.randint(0, 1))
        model_parameters['is_conv_bis'].append(is_conv_bis)
        
        if is_conv_bis == True:
            
            channels = random.randint(channels, NB_MAX_CHANNELS)
            kernel_conv = random.randint(kernel_conv, SIZE_MAX_KERNEL_CONV)
            if kernel_conv % 2 == 0:
                kernel_conv += 1
            if kernel_conv > dim:
                kernel_conv = dim
                if kernel_conv % 2 == 0:
                    kernel_conv -= 1
            stride_conv = random.randint(STRIDE_MIN_CONV, STRIDE_MAX_CONV)
            padding_conv = random.randint(PADDING_MIN_CONV, min(PADDING_MAX_CONV, int(kernel_conv/2)))

            model_parameters['channels'].append(channels) 
            model_parameters['kernel_conv'].append(kernel_conv)
            model_parameters['stride_conv'].append(stride_conv)
            model_parameters['padding_conv'].append(padding_conv)
            
            dim = int((dim - kernel_conv + 2 * padding_conv) / stride_conv) + 1
        
        is_pool_max = bool(random.randint(0, 1))
        kernel_pool = random.randint(SIZE_MIN_KERNEL_POOL, SIZE_MAX_KERNEL_POOL)
        if kernel_pool > dim:
            kernel_pool = dim
        stride_pool = random.randint(STRIDE_MIN_POOL, STRIDE_MAX_POOL)
        padding_pool = random.randint(PADDING_MIN_POOL, min(PADDING_MAX_POOL, int(kernel_pool/2)))
        
        model_parameters['is_pool_max'].append(is_pool_max) 
        model_parameters['kernel_pool'].append(kernel_pool) 
        model_parameters['stride_pool'].append(stride_pool) 
        model_parameters['padding_pool'].append(padding_pool) 
        
        dim = int((dim - kernel_pool + 2 * padding_pool) / stride_pool) + 1
        
        dropout_conv = random.random()
        batchnorm_conv = bool(random.randint(0, 1))
        
        model_parameters['dropout_conv'].append(dropout_conv)
        model_parameters['batchnorm_conv'].append(batchnorm_conv)
        
    model_parameters['size_linear_layer'].append(dim * dim * channels)
    
    for n_linear_layer in range(0, nb_linear_layers):
        
        size_linear_layer = int(loguniform.rvs(SIZE_MIN_LINEAR_LAYER, SIZE_MAX_LINEAR_LAYER, size=1)[0])
        dropout_linear = random.random()
        batchnorm_linear = bool(random.randint(0, 1))
        
        model_parameters['size_linear_layer'].append(size_linear_layer)
        model_parameters['dropout_linear'].append(dropout_linear)
        model_parameters['batchnorm_linear'].append(batchnorm_linear)
    
    return model_parameters