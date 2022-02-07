import numpy as np
from chessAI.preprocessing.gameMatrixMethod import create_game_matrix_first_method
from chessAI.preprocessing.pgnParsing import parse_pgn_to_list_board_rep

def create_game_matrices_one_game(pgn_text, is_white_win):
    
    # Get a list of board representation for each position in the game (except first and last)
    list_board_rep = parse_pgn_to_list_board_rep(pgn_text)
    
    list_matrices_white = []
    list_matrices_black = []
    is_white_move = True
            
    # Iterate over position
    for board_rep in list_board_rep:
        if is_white_move: list_matrices_white.append(create_game_matrix_first_method(board_rep))
        else: list_matrices_black.append(create_game_matrix_first_method(board_rep))
        is_white_move = not is_white_move
            
    # Concat all matrix in the list
    matrices_white = np.concatenate(list_matrices_white, axis=None)
    matrices_black = np.concatenate(list_matrices_black, axis=None)
    # Reshape
    matrices_white = matrices_white.reshape(len(list_matrices_white), 12, 8, 8)
    matrices_black = matrices_black.reshape(len(list_matrices_black), 12, 8, 8)
    
    # Create y matrix (is_white_win)
    y_white = np.full(matrices_white.shape[0], is_white_win, dtype=bool)
    y_black = np.full(matrices_black.shape[0], not is_white_win, dtype=bool)
        
    return matrices_white, matrices_black, y_white, y_black


def create_game_matrices_one_chunk_games(df_chunk):
        
    list_matrices_white = []
    list_matrices_black = []
    list_y_white, list_y_black = [], []
        
    # Iterate over games and add matrices to the corresponding list
    for index, row in df_chunk.iterrows():
            
        matrices_white_game, matrices_black_game, y_white_game, y_black_game = create_game_matrices_one_game(row.pgn_text, row.is_white_win)
        
        list_matrices_white.append(matrices_white_game)
        list_matrices_black.append(matrices_black_game)
        list_y_white.append(y_white_game)
        list_y_black.append(y_black_game)
            
    # Concat the list
    matrices_white = np.concatenate(list_matrices_white)
    matrices_black = np.concatenate(list_matrices_black)
    y_white = np.concatenate(list_y_white)
    y_black = np.concatenate(list_y_black)
        
    return matrices_white, matrices_black, y_white, y_black