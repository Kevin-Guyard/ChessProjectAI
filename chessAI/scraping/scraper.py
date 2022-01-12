import pandas as pd
import threading
import multiprocessing
from chessAI.scraping.playersData import get_list_players
from chessAI.scraping.gamesData import get_games_all_time_one_player

"""This module provides the Scraper class"""

class Scraper():
    
    def __init__(self):
        
        """Constructor method"""
        
        self._list_df_thread_1 = []
        self._list_df_thread_2 = []
        self._list_df_thread_3 = []
        self._list_df_thread_4 = []
        self._list_df_thread_5 = []
        self._list_df_thread_6 = []
        self._list_df_thread_7 = []
        self._list_df_thread_8 = []
        self._list_players = []
        self._nb_player_done = 0
        self._lock_list_player = threading.Lock()
        self._lock_nb_player_done = threading.Lock()
        
        
    def get_a_player(self):
        
        """Return a player to scrap
        
        :return actual_player: name of the player to scrap
        :rtype actual_player: string
        """
        
        # Use lock for thread safety
        with self._lock_list_player:
            # If there is at least another player to scrap in self._list_players, pop it, else None
            if len(self._list_players) > 0: 
                actual_player = self._list_players.pop(0)
            else:
                actual_player = None
                
        return actual_player
        
        
    def thread_get_games_all_time_all_players(self, elo_min, n_thread, nb_player):
        
        """Thread method for get_games_all_time_all_players. Do not use directly this method but use get_games_all_time_all_players.
        This threaded method pop players on self.list_player and get the games for this player, add it to its dataframe and iterate until there is no more player.
        
        :param elo_min: elo minimum for both player to keep a game
        :param elo_min: integer
        
        :param n_thread: thread number
        :type n_thread: integer
        
        :param nb_player: number of player in list of player (used to give achievement information)
        :type nb_player: integer
        """
        
        # Get a player
        actual_player = self.get_a_player()
        
        while actual_player != None:
            
            df_player = get_games_all_time_one_player(actual_player, elo_min)
            
            if df_player.shape[0] > 0:
                if n_thread == 1: self._df_thread_1 = self._list_df_thread_1.append(df_player)
                elif n_thread == 2: self._df_thread_2 = self._list_df_thread_2.append(df_player)
                elif n_thread == 3: self._df_thread_3 = self._list_df_thread_3.append(df_player)
                elif n_thread == 4: self._df_thread_4 = self._list_df_thread_4.append(df_player)
                elif n_thread == 5: self._df_thread_5 = self._list_df_thread_5.append(df_player)
                elif n_thread == 6: self._df_thread_6 = self._list_df_thread_6.append(df_player)
                elif n_thread == 7: self._df_thread_7 = self._list_df_thread_7.append(df_player)
                elif n_thread == 8: self._df_thread_8 = self._list_df_thread_8.append(df_player)
                                    
            with self._lock_nb_player_done:
                self._nb_player_done += 1
                print('Scraping: ' + str(self._nb_player_done) + '/' + str(nb_player) + ' players done', end='\r')
            
            # Get next player
            actual_player = self.get_a_player()
        
        
    def get_games_all_time_all_players(self, list_titles=['GM'], elo_min=2000):
        
        """Return all the games played by each player with the a title in the list of titles and where both player have an elo superior or equal to elo min
        
        :param list_titles: list of the titles for which get the games of each player with this title. Default: ['GM'] (GrandMaster)
        :type list_titles: list of string
        
        :param elo_min: elo minimum of both player on a game to keep the game. Default: 2000
        :type elo_min: integer
        
        :return df_global: dataframe with a row by game and 2 columns: is_white_win and pgn_text
        :rtype df_global: pd.DataFrame
        """
        
        # Get the list of players for each title given
        for title in list_titles: 
            self._list_players += get_list_players(title)
            
        nb_player = len(self._list_players)
        threads = []
        
        # Count the number of cpu for threading (max 8 thread)
        nb_core = multiprocessing.cpu_count()
        if nb_core > 8: nb_core = 8
        
        # Create and start thread to do the task
        for n_thread in range(1, nb_core+1):
            threads.append(threading.Thread(target=self.thread_get_games_all_time_all_players, args=(elo_min, n_thread, nb_player, )))
            threads[n_thread-1].start()
            
        for thread in threads:
            thread.join()
            
        # Concat data from each thread
        df_global = pd.concat([df for df in \
                               self._list_df_thread_1 + \
                               self._list_df_thread_2 + \
                               self._list_df_thread_3 + \
                               self._list_df_thread_4 + \
                               self._list_df_thread_5 + \
                               self._list_df_thread_6 + \
                               self._list_df_thread_7 + \
                               self._list_df_thread_8
                              ])
                
        # Drop duplicated rows on link field, drop link and reset index
        df_global.drop_duplicates(inplace=True, subset=['link'])
        df_global.drop(labels='link', axis=1, inplace=True)
        df_global.reset_index(drop=True, inplace=True)
        
        return df_global
