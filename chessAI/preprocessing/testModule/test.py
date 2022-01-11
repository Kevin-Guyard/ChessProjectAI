from chessAI.preprocessing.gameMatricesCreation import *
from chessAI.preprocessing.gameMatrixMethod import *
from chessAI.preprocessing.pgnParsing import *
from chessAI.preprocessing.testModule.convertion import *

def test_game_matrix_method_unitary(pgn_text):
    
    list_board_rep = parse_pgn_to_list_board_rep(pgn_text)
    board_rep = list_board_rep[5]

    matrix_first_method = create_game_matrix_first_method(board_rep)
    matrix_second_method = create_game_matrix_second_method(board_rep)
    matrix_third_method = create_game_matrix_third_method(board_rep)
    matrix_fourth_method = create_game_matrix_fourth_method(board_rep)

    board_rep_1 = convert_game_matrix_to_board_rep(matrix_first_method, 1)
    board_rep_2 = convert_game_matrix_to_board_rep(matrix_second_method, 2)
    board_rep_3 = convert_game_matrix_to_board_rep(matrix_third_method, 3)
    board_rep_4 = convert_game_matrix_to_board_rep(matrix_fourth_method, 4)
    
    return board_rep == board_rep_1 and board_rep == board_rep_2 and board_rep == board_rep_3 and board_rep == board_rep_4


def test_game_matrices_one_game(pgn_text):
    
    is_ok = True
    list_board_rep = parse_pgn_to_list_board_rep(pgn_text)
    board_rep = list_board_rep[5]

    list_matrices_white, list_matrices_black = create_game_matrices_one_game(pgn_text)
    list_list_board_rep_find = convert_list_matrices_to_board_rep(list_matrices_white, list_matrices_black)

    for list_board_rep_fin in list_list_board_rep_find:
        is_ok = is_ok and list_board_rep == list_board_rep_fin
        
    return is_ok