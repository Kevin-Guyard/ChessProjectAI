def get_list_model_parameters_LinearNN1(n_method):
    
    if n_method == 1: input_size = 8 * 8 * 12
    elif n_method == 2: input_size = 8 * 8 * 6
    elif n_method == 3: input_size = 8 * 8 * 4
    elif n_method == 4: input_size = 8 * 8 * 2
        
    model_parameters = []
    
    for hidden_size1 in [10, 100, 1000, 10000]:
        
        dic = {
            'model_name': 'LinearNN1', 
            'input_size': input_size, 
            'hidden_size1': hidden_size1, 
            'activation1': 'LeakyReLU', 
            'activation1_slope': 0.1
        }
        
        model_parameters.append(dic)
        
    return model_parameters