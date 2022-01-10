import pandas as pd
import requests
from datetime import date
import time
from chessAI.scraping.checkExtractedData import is_extracted_data_accepted
from chessAI.scraping.dataExtraction import extract_data_from_raw_text_game, convert_exctracted_data_to_dataframe
from chessAI.scraping.playersData import get_player_inscription_date

"""This module provides functions to get information on games

    - get_games_one_month_one_player: return the games played by a player during a specified monthly period
    - get_games_all_time_one_player: return the games played by a player (all time)
"""

def get_games_one_month_one_player(month, year, player, elo_min):
    
    """Return the games played by the player during the specified monthly period
    
    :param month: month of the monthly period
    :type month: integer
    
    :param year: year of the monthly period
    :type year: integer
    
    :param player: name of the player for who get the games
    :type player: string
    
    :param elo_min: elo minimum for both player to keep a game
    :type elo_min: integer
    
    :return df_month: dataframe with a row by game played and 3 columns: is_white_win, link and pgn_text
    :rtype df_month: pd.DataFrame
    """
    
    df_month = pd.DataFrame()
    
    # API url to request
    url = 'https://api.chess.com/pub/player/' + player + '/games/' + str(year) + '/' + str(month) + '/pgn'
    
    # Get request
    page_received = False
    while page_received == False:
        try:
            page_game_month = requests.get(url)
            page_received = True
        except:
            page_received = False
            time.sleep(0.01)
            
    # Handle case when server does not respond correctly (correct = code 200)
    while page_game_month.status_code != 200:
        time.sleep(0.01)
        page_game_month = requests.get(url)
    page_game_month_text = page_game_month.text
    
    # If the get request is not empty (i.e. the player have play at least one game in the month)
    if page_game_month_text != '':
        # Split the get respond by match data
        page_game_month_text_splited = page_game_month_text.split('\n\n\n')
        
        # Iterate over each match data
        for raw_text_data_game in page_game_month_text_splited:
            
            # Extract data, check validation and create a dataframe for the game, then concat with df_month
            event, result, white_elo, black_elo, termination, link, pgn_text = extract_data_from_raw_text_game(raw_text_data_game)
            if is_extracted_data_accepted(event, result, white_elo, black_elo, elo_min, termination, link, pgn_text) == True:
                df_match = convert_exctracted_data_to_dataframe(result, link, pgn_text)
                df_month = pd.concat([df_month, df_match]) 
                
        # Reset index
        df_month.reset_index(inplace=True, drop=True)
            
    return df_month


def get_games_all_time_one_player(player, elo_min):
    
    """Return the games played by the player during all time
    
    :param player: name of the player for who get the games
    :type player: string
    
    :param elo_min: elo minimum for both player to keep a game
    :type elo_min: integer
    
    :return df_player: dataframe with a row by game played and 3 columns: is_white_win, link and pgn_text
    :rtype df_player: pd.DataFrame
    """
    
    df_player = pd.DataFrame()
        
    # Get the month and the year actual
    month_today = date.today().month
    year_today = date.today().year
        
    # Get the inscription date of the player
    month_i, year_i = get_player_inscription_date(player)
        
    # Iterate from the date of inscription to today
    while (year_i < year_today) or ((month_i <= month_today) and (year_i == year_today)):
        df_month = get_games_one_month_one_player(month_i, year_i, player, elo_min)
        df_player = pd.concat([df_player, df_month])
        
        month_i += 1
        if month_i == 13:
            month_i = 1
            year_i += 1
                
    df_player.reset_index(inplace=True, drop=True)
        
    return df_player