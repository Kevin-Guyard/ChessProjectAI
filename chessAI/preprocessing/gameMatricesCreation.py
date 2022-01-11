import numpy as np
from chessAI.preprocessing.gameMatrixMethod import create_game_matrix_first_method, create_game_matrix_second_method, create_game_matrix_third_method, create_game_matrix_fourth_method
from chessAI.preprocessing.pgnParsing import parse_pgn_to_list_board_rep

"""This modules provides functions to create the game matrices for a game:
    
    - create_game_matrices_one_method_one_game: return the white and black game matrices for a game using one method
    - create_game_matrices_one_game: return the white and black game matrices for a game using 4 method
"""

def create_game_matrices_one_method_one_game(list_board_rep, n_method):
    
    """Return matrices of game position of all position of the game (except first and last) with the specified method (1 to 4)
    
    :param list_board_rep: list of the board representation of each position of the game
    :type list_board_rep: list of string
    
    :param n_method: number of the method (1 to 4) to use to create game matrices
    :type n_method: integer
    
    :return matrices_white: matrix of game matrices (1 matrix for one position in the game) for white move
    :rtype matrices_white: np.array of shape (x, 8, 8, y) (where x is the number of white position of the game, y the last dim shape depending of the choosen method), dtype=(bool if n_method=1, int if n_method=2, float if n_method=3 or 4)
    
    :return matrices_black: matrix of game matrices (1 matrix for one position in the game) for black move
    :rtype matrices_black: np.array of shape (x, 8, 8, y) (where x is the number of black position of the game, y the last dim shape depending of the choosen method), dtype=(bool if n_method=1, int if n_method=2, float if n_method=3 or 4)
    """
    
    list_matrices_white = []
    list_matrices_black = []
    is_white_move = True
    
    # Choose function and shape of the last dim according to the choosen method
    if n_method == 1:
        matrix_creation_method = create_game_matrix_first_method
        shape_last_dim = 12
    elif n_method == 2:
        matrix_creation_method = create_game_matrix_second_method
        shape_last_dim = 6
    elif n_method == 3:
        matrix_creation_method = create_game_matrix_third_method
        shape_last_dim = 4
    elif n_method == 4:
        matrix_creation_method = create_game_matrix_fourth_method
        shape_last_dim = 2
        
    # Iterate over position
    for board_rep in list_board_rep:
        if is_white_move: list_matrices_white.append(matrix_creation_method(board_rep))
        else: list_matrices_black.append(matrix_creation_method(board_rep))
        is_white_move = not is_white_move
            
    # Concat all matrix in the list
    matrices_white = np.concatenate(list_matrices_white, axis=None)
    matrices_black = np.concatenate(list_matrices_black, axis=None)
    # Reshape
    matrices_white = matrices_white.reshape(len(list_matrices_white), 8, 8, shape_last_dim)
    matrices_black = matrices_black.reshape(len(list_matrices_black), 8, 8, shape_last_dim)
    
    return matrices_white, matrices_black


def create_features_one_game(pgn_text, is_white_win):
    
    """Return matrices of game position of all position of the game (except first and last) with the 4 methods
    
    :param pgn_text: pgn of the game
    :type pgn_text: string
    
    :param is_white_win: True if the game is win by the white player else False
    :type is_white_win: boolean
    
    :return list_matrices_white: list of the matrix of game matrices (1 matrix for one position in the game) for white move (1 item in the list by method)
    :rtype list_matrices_white: list of np.array (length 4), np.array of shape (x, 8, 8, y) (where x is the number of white position of the game, y the last dim shape depending of the method), dtype=(bool for first method, int for second and float for thrid and fourth)
    
    :return list_matrices_black: list of the matrix of game matrices (1 matrix for one position in the game) for black move (1 item in the list by method)
    :rtype list_matrices_black: list of np.array (length 4), np.array of shape (x, 8, 8, y) (where x is the number of black position of the game, y the last dim shape depending of the method), dtype=(bool for first method, int for second and float for thrid and fourth)
    
    :return y_white: array with full of True or False depending of is_white_win
    :rtype y_white: np.array of shape (x) dtype=bool
    
    :return y_black: array with full of True or False depending of is_white_win
    :rtype y_black: np.array of shape (x) dtype=bool
    """
    
    list_matrices_white = []
    list_matrices_black = []
    
    # Get a list of board representation for each position in the game (except first and last)
    list_board_rep = parse_pgn_to_list_board_rep(pgn_text)
    
    for n_method in range(1, 5):
        matrices_white, matrices_black = create_game_matrices_one_method_one_game(list_board_rep, n_method)
        list_matrices_white.append(matrices_white)
        list_matrices_black.append(matrices_black)
        
    # Get white and black nb
    nb_move_white = list_matrices_white[0].shape[0]
    nb_move_black = list_matrices_black[0].shape[0]
    
    # Create y matrix (is_white_win)
    y_white = np.full(nb_move_white, is_white_win, dtype=bool)
    y_black = np.full(nb_move_black, not is_white_win, dtype=bool)
        
    return list_matrices_white, list_matrices_black, y_white, y_black