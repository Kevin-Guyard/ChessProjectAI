def get_shape_X_linear(n_method):
    
    if n_method == 1: shape_X = (8 * 8 * 12, )
    elif n_method == 2: shape_X = (8 * 8 * 6, )
    elif n_method == 3: shape_X = (8 * 8 * 4, )
    elif n_method == 4: shape_X = (8 * 8 * 2, )
        
    return shape_X