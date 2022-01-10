import requests
from datetime import datetime
import time

"""This module provides functions to get information on players

    - get_player_inscription_date: return the inscription date of the specified player
    - get_list_players: return the list of name of players with the title specified
"""

def get_player_inscription_date(player):
    
    """Return the inscription date of the specified player
    
    :param player: name of the player
    :type player: string
    
    :return inscription_month: month of the inscription of the player specified
    :rtype: integer
    
    :return inscription_year: year of the inscription of the player specified
    :rtype: integer
    """
    
    # API url to request
    url_player = 'https://api.chess.com/pub/player/' + player
    
    # Get request 
    page_received = False
    while page_received == False:
        try:
            page_player = requests.get(url_player)
            page_received = True
        except:
            page_received = False
            time.sleep(0.01)
            
    # Handle case when server does not respond correctly (correct = code 200)
    while page_player.status_code != 200:
        time.sleep(0.01)
        page_player = requests.get(url_player)
            
    # Clean the text to get the inscription date of the player in a timestamp format
    page_player_text = page_player.text
    inscription_date_timestamp = int(page_player_text.split('\"joined\":')[1].split(',')[0])
    
    # Convert timestamp to date then to month and year
    inscription_date = datetime.fromtimestamp(inscription_date_timestamp)
    inscription_month = int(inscription_date.month)
    inscription_year = int(inscription_date.year)
    
    return inscription_month, inscription_year


def get_list_players(title):
    
    """Return the list of name of players with the title specified
    
    :param title: title for which return the list of player
    :type title: string
    
    :return list_players: list of name of player with the title specified
    :rtype list_players: list of string
    """
        
    # API url to request
    url = 'https://api.chess.com/pub/titled/' + title
        
    # Get request and clean the text respond to obtain a list of player in the format "player1", "player2", ...
    page_list_players = requests.get(url)
    page_list_players_text = page_list_players.text
    list_players_unclean = page_list_players_text.split('[')[1].split(']')[0].split(',')
        
    list_players = []
        
    # Clean player name from ""
    for player in list_players_unclean:
        len_name = len(player)
        player = player[1:len_name-1]
        list_players.append(player)
            
    return list_players