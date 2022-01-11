import numpy as np

"""This modules provides functions to convert board representation (string) to game matrix (numpy array):
    
    - create_game_matrix_first_method: return the associated game matrix to the board representation according to the first method
    - create_game_matrix_second_method: return the associated game matrix to the board representation according to the second method
    - create_game_matrix_third_method: return the associated game matrix to the board representation according to the third method
    - create_game_matrix_fourth_method: return the associated game matrix to the board representation according to the fourth method
"""


def create_game_matrix_first_method(board_rep):
    
    """Return a numpy matrix which represent the board given in input.
    First method: a 8*8 matrix whith entries of dim 12. Only one of the entries can be non-zeros if there is a piece in the case. Other are 0
    dim 0: white pawn (1) / 1: black pawn (1) / 2: white knight (1) / 3: black knight (1) / 4: white bishop (1) / 5: black bishop (1) / 6: white rook (1) / 7: black rook (1) / 8: white queen (1) / 9: black queen (1) / 10: white king (1) / 11: black king (1)
    
    :param board_rep: board representation (list of string for which each string represents a line)
    :type board_rep: list of string of lenght 8
    
    :return game_matrix: game matrix which represent the actual position of the pieces
    :rtype game_matrix: np.array of shape (8, 8, 12), dtype=bool
    """
    
    # Create empty matrix
    game_matrix = np.zeros((8, 8, 12), dtype=bool)
    line = 0
    
    # Iterate over all line
    while line < 8:
        
        # Pop the last line representation of board representation (lines are inversed, board_rep start at line 8 and finish at line 1)
        line_rep = board_rep[7 - line]
        column = 0
        
        # Iterate over each character of the line representation
        for char in line_rep:
            # If the character is a number x, there is x cases empty at the actual position
            if char >= '1' and char <= '8':
                column += int(char)
            # Else find the piece and determine the dimension and the value to modify in the game matrix
            else:
                if char == 'P': dim = 0
                elif char == 'p': dim = 1
                elif char == 'N': dim = 2
                elif char == 'n': dim = 3
                elif char == 'B': dim = 4
                elif char == 'b': dim = 5
                elif char == 'R': dim = 6
                elif char == 'r': dim = 7
                elif char == 'Q': dim = 8
                elif char == 'q': dim = 9
                elif char == 'K': dim = 10
                elif char == 'k': dim = 11
                    
                game_matrix[line][column][dim] = 1
                column += 1
                
        line += 1
        
    return game_matrix


def create_game_matrix_second_method(board_rep):
    
    """Return a numpy matrix which represent the board given in input.
    Second method: a 8*8 matrix whith entries of dim 6. Only one of the entries can be non-zero if there is a piece in the case. Other are 0.
    dim 0: pawn (1=white/-1=black) / 1: knight (1=white/-1=black) / 2: bishop (1=white/-1=black) / 3: rook (1=white/-1=black) / 4: queen (1=white/-1=black) / 5: king (1=white/-1=black)
    
    :param board_rep: board representation (list of string for which each string represents a line)
    :type board_rep: list of string of lenght 8
    
    :return game_matrix: game matrix which represent the actual position of the pieces
    :rtype game_matrix: np.array of shape (8, 8, 6), dtype=int
    """
    
    # Create empty matrix
    game_matrix = np.zeros((8, 8, 6), dtype=int)
    line = 0
    
    # Iterate over all line
    while line < 8:
        
        # Pop the last line representation of board representation (lines are inversed, board_rep start at line 8 and finish at line 1)
        line_rep = board_rep[7 - line]
        column = 0
        
        # Iterate over each character of the line representation
        for char in line_rep:
            # If the character is a number x, there is x cases empty at the actual position
            if char >= '1' and char <= '8':
                column += int(char)
            # Else find the piece and determine the dimension and the value to modify in the game matrix
            else:
                if char == 'P': dim, val = 0, 1
                elif char == 'p': dim, val = 0, -1
                elif char == 'N': dim, val = 1, 1
                elif char == 'n': dim, val = 1, -1
                elif char == 'B': dim, val = 2, 1
                elif char == 'b': dim, val = 2, -1
                elif char == 'R': dim, val = 3, 1
                elif char == 'r': dim, val = 3, -1
                elif char == 'Q': dim, val = 4, 1
                elif char == 'q': dim, val = 4, -1
                elif char == 'K': dim, val = 5, 1
                elif char == 'k': dim, val = 5, -1
                    
                game_matrix[line][column][dim] = val
                column += 1
            
        line += 1
        
    return game_matrix


def create_game_matrix_third_method(board_rep):
    
    """Return a numpy matrix which represent the board given in input.
    Third method: a 8*8 matrix whith entries of dim 4. Only one of the entries can be non-zero if there is a piece in the case. Other are 0.
    
    dim 0: white pawn (1.0), knight (3.2), bishop (3.33), rook (5.1), queen (8.8) / 1: black pawn (1.0), knight (3.2), bishop (3.33), rook (5.1), queen (8.8) / 2: white king (1) / 3: black king (1)
    
    :param board_rep: board representation (list of string for which each string represents a line)
    :type board_rep: list of string of lenght 8
    
    :return game_matrix: game matrix which represent the actual position of the pieces
    :rtype game_matrix: np.array of shape (8, 8, 4), dtype=float
    """
    
    # Create empty matrix
    game_matrix = np.zeros((8, 8, 4), dtype=float)
    line = 0
    
    # Iterate over all line
    while line < 8:
        
        # Pop the last line representation of board representation (lines are inversed, board_rep start at line 8 and finish at line 1)
        line_rep = board_rep[7 - line]
        column = 0
        
        # Iterate over each character of the line representation
        for char in line_rep:
            # If the character is a number x, there is x cases empty at the actual position
            if char >= '1' and char <= '8':
                column += int(char)
            # Else find the piece and determine the dimension and the value to modify in the game matrix
            else:
                if char == 'P': dim, val = 0, 1.0
                elif char == 'p': dim, val = 1, 1.0
                elif char == 'N': dim, val = 0, 3.2
                elif char == 'n': dim, val = 1, 3.2
                elif char == 'B': dim, val = 0, 3.33
                elif char == 'b': dim, val = 1, 3.33
                elif char == 'R': dim, val = 0, 5.1
                elif char == 'r': dim, val = 1, 5.1
                elif char == 'Q': dim, val = 0, 8.8
                elif char == 'q': dim, val = 1, 8.8
                elif char == 'K': dim, val = 2, 1
                elif char == 'k': dim, val = 3, 1
                    
                game_matrix[line][column][dim] = val
                column += 1
            
        line += 1
        
    return game_matrix


def create_game_matrix_fourth_method(board_rep):
    
    """Return a numpy matrix which represent the board given in input.
    Fourth method: a 8*8 matrix whith entries of dim 4. Only one of the entries can be non-zero if there is a piece in the case. Other are 0.
    
    dim 0: white pawn (1.0=white/-1.0=black), knight (3.2=white/-3.2=black), bishop (3.33=white/-3.33=black), rook (5.1=white/-5.1=black), queen (8.8=white/-8.8=black) / 1: king(1=white/-1=black)
    
    :param board_rep: board representation (list of string for which each string represents a line)
    :type board_rep: list of string of lenght 8
    
    :return game_matrix: game matrix which represent the actual position of the pieces
    :rtype game_matrix: np.array of shape (8, 8, 2), dtype=float
    """
    
    # Create empty matrix
    game_matrix = np.zeros((8, 8, 2), dtype=float)
    line = 0
    
    # Iterate over all line
    while line < 8:
        
        # Pop the last line representation of board representation (lines are inversed, board_rep start at line 8 and finish at line 1)
        line_rep = board_rep[7 - line]
        column = 0
        
        # Iterate over each character of the line representation
        for char in line_rep:
            # If the character is a number x, there is x cases empty at the actual position
            if char >= '1' and char <= '8':
                column += int(char)
            # Else find the piece and determine the dimension and the value to modify in the game matrix
            else:
                if char == 'P': dim, val = 0, 1.0
                elif char == 'p': dim, val = 0, -1.0
                elif char == 'N': dim, val = 0, 3.2
                elif char == 'n': dim, val = 0, -3.2
                elif char == 'B': dim, val = 0, 3.33
                elif char == 'b': dim, val = 0, -3.33
                elif char == 'R': dim, val = 0, 5.1
                elif char == 'r': dim, val = 0, -5.1
                elif char == 'Q': dim, val = 0, 8.8
                elif char == 'q': dim, val = 0, -8.8
                elif char == 'K': dim, val = 1, 1
                elif char == 'k': dim, val = 1, -1
                    
                game_matrix[line][column][dim] = val
                column += 1
            
        line += 1
        
    return game_matrix