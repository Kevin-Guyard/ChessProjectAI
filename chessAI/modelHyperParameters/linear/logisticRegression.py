def get_list_model_parameters_LogisticRegression(n_method):
    
    if n_method == 1: input_size = 8 * 8 * 12
    elif n_method == 2: input_size = 8 * 8 * 6
    elif n_method == 3: input_size = 8 * 8 * 4
    elif n_method == 4: input_size = 8 * 8 * 2
       
    model_parameters = []
    
    dic = {
        'model_name': 'LogisticRegression', 
        'input_size': input_size
    }
    
    model_parameters.append(dic)
    
    return model_parameters