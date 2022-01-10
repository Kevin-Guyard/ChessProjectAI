import io
import chess.pgn
import logging

"""This module provides functions to test extracted data get by scraping the chess.com API

    - is_extracted_data_accepted: return True if extracted data are correct and accepted
    - is_pgn_format_correct: return True if pgn does not contains errors
"""


def is_extracted_data_accepted(event, result, white_elo, black_elo, elo_min, termination, link, pgn_text):
    
    """Determine if the extracted data is accepted (i.e. event is Live Chess, result is a clear victory, white and black elo are superior or equal to elo min, termination is checkmate and pgn does not contains errors)
    
    :param event: event of the game
    :type event: string
    
    :param result: result of the game
    :type result: string
    
    :param white_elo: elo of the white player
    :type white_elo: integer
    
    :param black_elo: elo of the black player
    :type black_elo: integer
    
    :param elo_min: elo minimum for both player to accepte the game
    :type elo_min: integer
    
    :param termination: termination of the game
    :type termination: string
    
    :param link: link of the game
    :type link: string
    
    :param pgn_text: pgn text data of the moves of the games
    :type pgn_text: string
    
    :return data_accepted: True if data is accepted else False
    :rtype: boolean
    """
    
    data_accepted = True
    
    if event != 'Live Chess': data_accepted = False
    elif result not in ['1-0', '0-1']: data_accepted = False
    elif white_elo < elo_min or black_elo < elo_min: data_accepted = False
    elif not ' won by checkmate' in termination: data_accepted = False
    elif not is_pgn_format_correct(pgn_text): data_accepted = False
        
    return data_accepted


def is_pgn_format_correct(pgn_text):
    
    """Determine if the given pgn is correct (no illegal moves, no ambigius moves ...)
    
    :param pgn_text: pgn text data of the moves of the games
    :type pgn_text: string
    
    :return pgn_correct: True if pgn is correct else False
    :rtype pgn_correct: boolean
    """
    
    # Disable chess.pgn error on standart output
    logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)
    
    pgn = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn)
    nb_errors_in_pgn = len(game.errors)
    
    if nb_errors_in_pgn == 0: pgn_correct = True
    else: pgn_correct = False

    return pgn_correct  