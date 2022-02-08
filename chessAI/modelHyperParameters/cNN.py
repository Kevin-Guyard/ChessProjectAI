import random

NB_MAX_CHANNELS = 48
SIZE_MAX_KERNEL_CONV = 6 #(+1)
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

def get_model_parameters_CNN(model_name):
    
    nb_conv_layers = int(model_name.split('CNN')[1].split('-')[0])
    nb_linear_layers = int(model_name.split('CNN')[1].split('-')[1])
    
    model_parameters = {
        'model_name': model_name,
        'nb_conv_layers': nb_conv_layers,
        'nb_linear_layers': nb_linear_layers,
        'channels_0': 12
    }
    
    channels_last = 12
    kernel_conv_last = 1  
    dim_last = 8
    
    for n_conv_layer in range(0, nb_conv_layers):
        
        channels = random.randint(channels_last, NB_MAX_CHANNELS)
        kernel_conv = random.randint(kernel_conv_last, min(dim_last, SIZE_MAX_KERNEL_CONV))
        if kernel_conv % 2 == 0:
            kernel_conv += 1
        stride_conv = random.randint(STRIDE_MIN_CONV, STRIDE_MAX_CONV)
        padding_conv = random.randint(PADDING_MIN_CONV, min(PADDING_MAX_CONV, int(kernel_conv/2)))
        dim_conv = int((dim_last - kernel_conv + 2 * padding_conv) / stride_conv) + 1
        is_pool_max = bool(random.randint(0, 1))
        kernel_pool = random.randint(SIZE_MIN_KERNEL_POOL, min(dim_conv, SIZE_MAX_KERNEL_POOL))
        stride_pool = random.randint(STRIDE_MIN_POOL, STRIDE_MAX_POOL)
        padding_pool = random.randint(PADDING_MIN_POOL, min(PADDING_MAX_POOL, int(kernel_pool/2)))
        dim_pool = int((dim_conv - kernel_pool + 2 * padding_pool) / stride_pool) + 1
        dropout = random.random()
        batchnorm = bool(random.randint(0, 1))
        
        model_parameters['channels_' + str(n_conv_layer + 1)] = channels
        model_parameters['kernel_conv_' + str(n_conv_layer)] = kernel_conv
        model_parameters['stride_conv_' + str(n_conv_layer)] = stride_conv
        model_parameters['padding_conv_' + str(n_conv_layer)] = padding_conv
        model_parameters['is_pool_max_' + str(n_conv_layer)] = is_pool_max
        model_parameters['kernel_pool_' + str(n_conv_layer)] = kernel_pool
        model_parameters['stride_pool_' + str(n_conv_layer)] = stride_pool
        model_parameters['padding_pool_' + str(n_conv_layer)] = padding_pool
        model_parameters['dropout_' + str(n_conv_layer)] = dropout
        model_parameters['batchnorm_' + str(n_conv_layer)] = batchnorm
        
        channels_last = channels
        kernel_conv_last = kernel_conv
        dim_last = dim_pool
        
    model_parameters['linear'] = dim_pool * dim_pool * channels
            
    return model_parameters