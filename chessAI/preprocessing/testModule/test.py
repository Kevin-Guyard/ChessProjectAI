from chessAI.preprocessing.gameMatricesCreation import *
from chessAI.preprocessing.gameMatrixMethod import *
from chessAI.preprocessing.pgnParsing import *
from chessAI.preprocessing.preprocesser import *
from chessAI.preprocessing.testModule.convertion import *
import pandas as pd

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


def test_preprocesser(df):
    
    preprocesser = Preprocesser(df.copy())
    list_board_rep_white = []
    list_board_rep_black = []
    list_white_win = []
    list_black_win = []

    list_board_rep_white_1 = []
    list_board_rep_black_1 = []
    list_board_rep_white_2 = []
    list_board_rep_black_2 = []
    list_board_rep_white_3 = []
    list_board_rep_black_3 = []
    list_board_rep_white_4 = []
    list_board_rep_black_4 = []
    list_y_white = []
    list_y_black = []

    matrices_white_1, matrices_black_1, matrices_white_2, matrices_black_2, \
    matrices_white_3, matrices_black_3, matrices_white_4, matrices_black_4, \
    y_white, y_black = preprocesser.create_game_matrices_all_games(chunk_size=10)

    for index, row in df.iterrows():

        list_board_rep = parse_pgn_to_list_board_rep(row.pgn_text)
        is_white_move = True
        for board_rep in list_board_rep:
            if is_white_move: 
                list_board_rep_white.append(board_rep)
                list_white_win.append(row.is_white_win)
            else: 
                list_board_rep_black.append(board_rep)
                list_black_win.append(not row.is_white_win)
            is_white_move = not is_white_move    

    for matrix in matrices_white_1:
        list_board_rep_white_1.append(convert_game_matrix_to_board_rep(matrix, 1))

    for matrix in matrices_black_1:
        list_board_rep_black_1.append(convert_game_matrix_to_board_rep(matrix, 1))

    for matrix in matrices_white_2:
        list_board_rep_white_2.append(convert_game_matrix_to_board_rep(matrix, 2))

    for matrix in matrices_black_2:
        list_board_rep_black_2.append(convert_game_matrix_to_board_rep(matrix, 2))

    for matrix in matrices_white_3:
        list_board_rep_white_3.append(convert_game_matrix_to_board_rep(matrix, 3))

    for matrix in matrices_black_3:
        list_board_rep_black_3.append(convert_game_matrix_to_board_rep(matrix, 3))

    for matrix in matrices_white_4:
        list_board_rep_white_4.append(convert_game_matrix_to_board_rep(matrix, 4))

    for matrix in matrices_black_4:
        list_board_rep_black_4.append(convert_game_matrix_to_board_rep(matrix, 4))

    for y in y_white:
        list_y_white.append(y)

    for y in y_black:
        list_y_black.append(y)

    df_white_1 = pd.DataFrame({'board_rep': list_board_rep_white_1, 'is_white_win': list_y_white})
    df_white_2 = pd.DataFrame({'board_rep': list_board_rep_white_2, 'is_white_win': list_y_white})
    df_white_3 = pd.DataFrame({'board_rep': list_board_rep_white_3, 'is_white_win': list_y_white})
    df_white_4 = pd.DataFrame({'board_rep': list_board_rep_white_4, 'is_white_win': list_y_white})
    df_black_1 = pd.DataFrame({'board_rep': list_board_rep_black_1, 'is_white_win': list_y_black})
    df_black_2 = pd.DataFrame({'board_rep': list_board_rep_black_2, 'is_white_win': list_y_black})
    df_black_3 = pd.DataFrame({'board_rep': list_board_rep_black_3, 'is_white_win': list_y_black})
    df_black_4 = pd.DataFrame({'board_rep': list_board_rep_black_4, 'is_white_win': list_y_black})
    df_white = pd.DataFrame({'board_rep': list_board_rep_white, 'is_white_win': list_white_win})
    df_black = pd.DataFrame({'board_rep': list_board_rep_black, 'is_white_win': list_black_win})

    df_white_1.board_rep = df_white_1.board_rep.apply(lambda x: str(x))
    df_white_2.board_rep = df_white_2.board_rep.apply(lambda x: str(x))
    df_white_3.board_rep = df_white_3.board_rep.apply(lambda x: str(x))
    df_white_4.board_rep = df_white_4.board_rep.apply(lambda x: str(x))
    df_black_1.board_rep = df_black_1.board_rep.apply(lambda x: str(x))
    df_black_2.board_rep = df_black_2.board_rep.apply(lambda x: str(x))
    df_black_3.board_rep = df_black_3.board_rep.apply(lambda x: str(x))
    df_black_4.board_rep = df_black_4.board_rep.apply(lambda x: str(x))
    df_white.board_rep = df_white.board_rep.apply(lambda x: str(x))
    df_black.board_rep = df_black.board_rep.apply(lambda x: str(x))

    df_white_1.drop_duplicates(inplace=True)
    df_white_2.drop_duplicates(inplace=True)
    df_white_3.drop_duplicates(inplace=True)
    df_white_4.drop_duplicates(inplace=True)
    df_black_1.drop_duplicates(inplace=True)
    df_black_2.drop_duplicates(inplace=True)
    df_black_3.drop_duplicates(inplace=True)
    df_black_4.drop_duplicates(inplace=True)
    df_white.drop_duplicates(inplace=True)
    df_black.drop_duplicates(inplace=True)

    return \
    df_white_1.shape == pd.concat([df_white, df_white_1]).drop_duplicates().shape and \
    df_white_2.shape == pd.concat([df_white, df_white_2]).drop_duplicates().shape and \
    df_white_3.shape == pd.concat([df_white, df_white_3]).drop_duplicates().shape and \
    df_white_4.shape == pd.concat([df_white, df_white_4]).drop_duplicates().shape and \
    df_black_1.shape == pd.concat([df_black, df_black_1]).drop_duplicates().shape and \
    df_black_2.shape == pd.concat([df_black, df_black_2]).drop_duplicates().shape and \
    df_black_3.shape == pd.concat([df_black, df_black_3]).drop_duplicates().shape and \
    df_black_4.shape == pd.concat([df_black, df_black_4]).drop_duplicates().shape