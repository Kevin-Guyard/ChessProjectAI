import numpy as np

def create_game_matrix_first_method(board_rep):
    
    # Create empty matrix
    game_matrix = np.zeros((12, 8, 8), dtype=bool)
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
                    
                game_matrix[dim][line][column] = 1
                column += 1
                
        line += 1
        
    return game_matrix