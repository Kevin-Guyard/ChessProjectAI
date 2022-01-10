import pandas as pd

"""This modules provide functions to extract data from raw text data get by scraping the chess.com API

    - extract_data_from_raw_text_game: return extracted data
    - convert_exctracted_data_to_dataframe: return a pandas dataframe from the extracted data
"""

def extract_data_from_raw_text_game(raw_text_data_game):
    
    """Extract the data from the raw text data game
    
    :param raw_text_data_game: raw text data game get by scraping chess.com API ([Event "xxx"], ['Site "xxx"'], ...)
    :type raw_text_data_game: string
    
    :return event: event of the game
    :rtype event: string
    
    :return result: result of the game
    :rtype result: string
    
    :return white_elo: elo of the white player
    :rtype white_elo: integer
    
    :return black_elo: elo of the black player
    :rtype black_elo: integer
    
    :return termination: termination of the game
    :rtype termination: string
    
    :return link: link of the game
    :rtype link: string
    
    :return pgn_text: pgn text data of the moves of the games
    :rtype pgn_text: string
    """
    
    # Use try/execpt in case missing filed in raw data
    try:    
        # Raw data are on format [Event "xxx"], ['Site "xxx"'], ...
        event = raw_text_data_game.split('[Event ')[1].split('"')[1]
        result = raw_text_data_game.split('[Result ')[1].split('"')[1]
        white_elo = int(raw_text_data_game.split('[WhiteElo ')[1].split('"')[1])
        black_elo = int(raw_text_data_game.split('[BlackElo ')[1].split('"')[1])
        termination = raw_text_data_game.split('[Termination ')[1].split('"')[1]
        link = raw_text_data_game.split('[Link ')[1].split('"')[1]
        pgn_text = raw_text_data_game.split('\n\n')[1]
        
    # This error occur if a field is missing in raw data (error occur by index an empty list after spliting)
    except IndexError:
        event = None
        result = None
        white_elo = None
        black_elo = None
        termination = None
        link = None
        pgn_text = None
        
    return event, result, white_elo, black_elo, termination, link, pgn_text


def convert_exctracted_data_to_dataframe(result, link, pgn_text):
    
    """Convert the extracted data from a game to a dataframe
    
    :param result: result of the game
    :type result: string
    
    :param link: link of the game
    :type link: string
    
    :param pgn_text: pgn text data of the moves of the games
    :type pgn_text: string
    
    :return df_game: dataframe with 1 row and 3 columns: is_white_win, link and pgn_text
    :rtype df_game: pd.DataFrame
    """

    if result == '1-0': is_white_win = True
    else: is_white_win = False
    
    df_game = pd.DataFrame({'is_white_win': is_white_win, 'link': link, 'pgn_text': pgn_text}, index=[0])
    df_game.is_white_win = df_game.is_white_win.astype(bool)

    return df_game