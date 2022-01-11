import io
import chess.pgn

"""This module provide a function to parse pgn:

    - parse_pgn_to_list_board_rep: return a list of board representation associated to all position of the given game pgn (except first and last position)
"""

def parse_pgn_to_list_board_rep(pgn_text):
    
    """Take a pgn in string format and return a list with one item for each position (except first and last position that are dropped).
    The item is itself a list of lenght 8. The first element of this list represent the line 7 of the chess board, the second the line 6 etc.
    The item start by describing first column. If it is a letter, then there is a piece in the case and continue to the next column. If there is a number x, then there is x empty case including the actual case and continue.
    
    :param pgn_text: pgn of the game
    :type pgn_text: string
    
    return list_board_rep: list of the board representation of each position of the game
    rtype list_board_rep: list of list of string
    """
    
    list_board_rep = []
    
    # Create an object game and an object board with the pgn
    pgn = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    
    # Iterate over moves of the game
    for move in game.mainline_moves():
        board.push(move)
        board_rep = board.__repr__().split('\'')[1].split(' ')[0].split('/')
        list_board_rep.append(board_rep)
    
    # Drop last one (game are assumed as win on a checkmate and the last position is the checkmate position thus we drop it)
    list_board_rep.pop()
    
    return list_board_rep