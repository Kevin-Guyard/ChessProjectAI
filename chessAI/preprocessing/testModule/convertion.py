import numpy as np

"""

    - convert_game_matrix_first_method_to_board_rep_matrix
    - convert_game_matrix_second_method_to_board_rep_matrix
    - convert_game_matrix_third_method_to_board_rep_matrix
    - convert_game_matrix_fourth_method_to_board_rep_matrix
    - convert_board_rep_matrix_to_board_rep
    - convert_list_matrices_to_board_rep

"""

def convert_game_matrix_to_board_rep(game_matrix, n_method):
    
    if n_method == 1:
        convertion_method = convert_game_matrix_first_method_to_board_rep_matrix
    elif n_method == 2:
        convertion_method = convert_game_matrix_second_method_to_board_rep_matrix
    elif n_method == 3:
        convertion_method = convert_game_matrix_third_method_to_board_rep_matrix
    elif n_method == 4:
        convertion_method = convert_game_matrix_fourth_method_to_board_rep_matrix
        
    board_rep_matrix = convertion_method(game_matrix)
    board_rep = convert_board_rep_matrix_to_board_rep(board_rep_matrix)
    
    return board_rep


def convert_game_matrix_first_method_to_board_rep_matrix(game_matrix):
    
    board_rep_matrix = np.zeros((8, 8), dtype=int)
    
    for line in range(0, 8):
        for column in range(0, 8):
            
            if game_matrix[line][column][0] == 1:
                board_rep_matrix[line][column] = 1
            elif game_matrix[line][column][1] == 1:
                board_rep_matrix[line][column] = 2
            elif game_matrix[line][column][2] == 1:
                board_rep_matrix[line][column] = 3
            elif game_matrix[line][column][3] == 1:
                board_rep_matrix[line][column] = 4
            elif game_matrix[line][column][4] == 1:
                board_rep_matrix[line][column] = 5
            elif game_matrix[line][column][5] == 1:
                board_rep_matrix[line][column] = 6
            elif game_matrix[line][column][6] == 1:
                board_rep_matrix[line][column] = 7
            elif game_matrix[line][column][7] == 1:
                board_rep_matrix[line][column] = 8
            elif game_matrix[line][column][8] == 1:
                board_rep_matrix[line][column] = 9
            elif game_matrix[line][column][9] == 1:
                board_rep_matrix[line][column] = 10
            elif game_matrix[line][column][10] == 1:
                board_rep_matrix[line][column] = 11
            elif game_matrix[line][column][11] == 1:
                board_rep_matrix[line][column] = 12
                
    return board_rep_matrix


def convert_game_matrix_second_method_to_board_rep_matrix(game_matrix):
    
    board_rep_matrix = np.zeros((8, 8), dtype=int)
    
    for line in range(0, 8):
        for column in range(0, 8):
            
            if game_matrix[line][column][0] == 1:
                board_rep_matrix[line][column] = 1
            elif game_matrix[line][column][0] == -1:
                board_rep_matrix[line][column] = 2
            elif game_matrix[line][column][1] == 1:
                board_rep_matrix[line][column] = 3
            elif game_matrix[line][column][1] == -1:
                board_rep_matrix[line][column] = 4
            elif game_matrix[line][column][2] == 1:
                board_rep_matrix[line][column] = 5
            elif game_matrix[line][column][2] == -1:
                board_rep_matrix[line][column] = 6
            elif game_matrix[line][column][3] == 1:
                board_rep_matrix[line][column] = 7
            elif game_matrix[line][column][3] == -1:
                board_rep_matrix[line][column] = 8
            elif game_matrix[line][column][4] == 1:
                board_rep_matrix[line][column] = 9
            elif game_matrix[line][column][4] == -1:
                board_rep_matrix[line][column] = 10
            elif game_matrix[line][column][5] == 1:
                board_rep_matrix[line][column] = 11
            elif game_matrix[line][column][5] == -1:
                board_rep_matrix[line][column] = 12
                
    return board_rep_matrix


def convert_game_matrix_third_method_to_board_rep_matrix(game_matrix):
    
    board_rep_matrix = np.zeros((8, 8), dtype=int)
    
    for line in range(0, 8):
        for column in range(0, 8):
            
            if game_matrix[line][column][0] == 1.0:
                board_rep_matrix[line][column] = 1
            elif game_matrix[line][column][1] == 1.0:
                board_rep_matrix[line][column] = 2
            elif game_matrix[line][column][0] == 3.2:
                board_rep_matrix[line][column] = 3
            elif game_matrix[line][column][1] == 3.2:
                board_rep_matrix[line][column] = 4
            elif game_matrix[line][column][0] == 3.33:
                board_rep_matrix[line][column] = 5
            elif game_matrix[line][column][1] == 3.33:
                board_rep_matrix[line][column] = 6
            elif game_matrix[line][column][0] == 5.1:
                board_rep_matrix[line][column] = 7
            elif game_matrix[line][column][1] == 5.1:
                board_rep_matrix[line][column] = 8
            elif game_matrix[line][column][0] == 8.8:
                board_rep_matrix[line][column] = 9
            elif game_matrix[line][column][1] == 8.8:
                board_rep_matrix[line][column] = 10
            elif game_matrix[line][column][2] == 1:
                board_rep_matrix[line][column] = 11
            elif game_matrix[line][column][3] == 1:
                board_rep_matrix[line][column] = 12
                
    return board_rep_matrix


def convert_game_matrix_fourth_method_to_board_rep_matrix(game_matrix):
    
    board_rep_matrix = np.zeros((8, 8), dtype=int)
    
    for line in range(0, 8):
        for column in range(0, 8):
            
            if game_matrix[line][column][0] == 1.0:
                board_rep_matrix[line][column] = 1
            elif game_matrix[line][column][0] == -1.0:
                board_rep_matrix[line][column] = 2
            elif game_matrix[line][column][0] == 3.2:
                board_rep_matrix[line][column] = 3
            elif game_matrix[line][column][0] == -3.2:
                board_rep_matrix[line][column] = 4
            elif game_matrix[line][column][0] == 3.33:
                board_rep_matrix[line][column] = 5
            elif game_matrix[line][column][0] == -3.33:
                board_rep_matrix[line][column] = 6
            elif game_matrix[line][column][0] == 5.1:
                board_rep_matrix[line][column] = 7
            elif game_matrix[line][column][0] == -5.1:
                board_rep_matrix[line][column] = 8
            elif game_matrix[line][column][0] == 8.8:
                board_rep_matrix[line][column] = 9
            elif game_matrix[line][column][0] == -8.8:
                board_rep_matrix[line][column] = 10
            elif game_matrix[line][column][1] == 1:
                board_rep_matrix[line][column] = 11
            elif game_matrix[line][column][1] == -1:
                board_rep_matrix[line][column] = 12
                
    return board_rep_matrix


def convert_board_rep_matrix_to_board_rep(board_rep_matrix):
    
    board_rep = []
    
    for line in range(7, -1, -1):
        line_rep = ''
        n_blank = 0
        for column in range(0, 8):
            if board_rep_matrix[line][column] == 0:
                n_blank += 1
            else:
                if n_blank != 0:
                    line_rep += str(n_blank)
                    n_blank = 0
                if board_rep_matrix[line][column] == 1:
                    line_rep += 'P'
                elif board_rep_matrix[line][column] == 2:
                    line_rep += 'p'
                elif board_rep_matrix[line][column] == 3:
                    line_rep += 'N'
                elif board_rep_matrix[line][column] == 4:
                    line_rep += 'n'
                elif board_rep_matrix[line][column] == 5:
                    line_rep += 'B'
                elif board_rep_matrix[line][column] == 6:
                    line_rep += 'b'
                elif board_rep_matrix[line][column] == 7:
                    line_rep += 'R'
                elif board_rep_matrix[line][column] == 8:
                    line_rep += 'r'
                elif board_rep_matrix[line][column] == 9:
                    line_rep += 'Q'
                elif board_rep_matrix[line][column] == 10:
                    line_rep += 'q'
                elif board_rep_matrix[line][column] == 11:
                    line_rep += 'K'
                elif board_rep_matrix[line][column] == 12:
                    line_rep += 'k'
            if column == 7:
                if n_blank != 0:
                    line_rep += str(n_blank)
                board_rep.append(line_rep)
                    
                        
    return board_rep


def convert_list_matrices_to_board_rep(list_matrices_white, list_matrices_black):
    
    list_board_rep = []
    
    for n_method in range(1, 5):
        
        list_board_rep_white = []
        list_board_rep_black = []
        list_board_rep_method = []
        
        for matrix in list_matrices_white[n_method - 1]:
            list_board_rep_white.append(convert_game_matrix_to_board_rep(matrix, n_method))
        for matrix in list_matrices_black[n_method - 1]:
            list_board_rep_black.append(convert_game_matrix_to_board_rep(matrix, n_method))
            
        while len(list_board_rep_white) > 0 and len(list_board_rep_black) > 0:
            list_board_rep_method.append(list_board_rep_white.pop(0))
            list_board_rep_method.append(list_board_rep_black.pop(0))
            
        if len(list_board_rep_white) > 0:
            list_board_rep_method.append(list_board_rep_white.pop(0))
            
        list_board_rep.append(list_board_rep_method)
    
    return list_board_rep