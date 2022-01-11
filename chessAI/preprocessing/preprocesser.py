import pandas as pd
import numpy as np
import threading
import multiprocessing
from chessAI.preprocessing.gameMatricesCreation import create_game_matrices_one_chunk_games

"""This module provides the Preprocesser class"""

class Preprocesser():
    
    """Preprocess data from the scraper
    
    :param df_games: dataframe from the scraper with columns pgn_text and is_white_win
    :type df_games: pd.DataFrame()
    """
    
    def __init__(self, df_games):
        
        """Constructor method"""
        
        self._df_games = df_games
        self._nb_games_done = 0
        self._list_matrices_white_1 = []
        self._list_matrices_white_2 = []
        self._list_matrices_white_3 = []
        self._list_matrices_white_4 = []
        self._list_matrices_black_1 = []
        self._list_matrices_black_2 = []
        self._list_matrices_black_3 = []
        self._list_matrices_black_4 = []
        self._list_y_white = []
        self._list_y_black = []
        self._lock_df_games = threading.Lock()
        self._lock_list = threading.Lock()
        self._lock_nb_games_done = threading.Lock()
        
        
    def get_a_chunk(self, chunk_size):
        
        """Return a chunk of self._df_games and drop it from the original dataframe. The size of the chunk is :
            - chunk_size if the size of the df is superior or equal to chunk_size
            - the size of the df if its size is inferior but not 0
            - an empty df else
        :param chunk_size: size of the chunk
        :type chunk_size: integer
        
        :return df_chunk: a chunk of self._df_games
        :rtype df_chunk: pd.DataFrame()
        """
        
        # Use lock to ensure thread safety access to the original df
        with self._lock_df_games:
            # If df is empty, df_chunk is empty
            if self._df_games.shape[0] == 0: 
                df_chunk = pd.DataFrame()
            else:
                # If the size of df is superior or equal to chunk_size, df_chunk is the last chunk_size row and drop it from original dataframe
                if self._df_games.shape[0] >= chunk_size:
                    df_chunk = self._df_games[self._df_games.shape[0] - chunk_size:self._df_games.shape[0]].copy()
                    self._df_games.drop(index=[x for x in range(self._df_games.shape[0] - chunk_size, self._df_games.shape[0])], inplace=True)
                # Else df_chunk is the rest of the original df
                else:
                    df_chunk = self._df_games[0:self._df_games.shape[0]].copy()
                    self._df_games.drop(index=[x for x in range(0, self._df_games.shape[0])], inplace=True)
                    
        return df_chunk
        
        
    def thread_create_game_matrices_all_game(self, chunk_size, nb_games):
        
        """Thread method for create_game_matrices_all_games. Do not use directly this method but use create_game_matrices_all_games.
        
        :param chunk_size: size of the chunks of dataframe to preprocess. If memory problems, decrease this number. Default: 100
        :type chunk_size: integer
        
        :param nb_games: number of game in the dataframe self._df_games
        :type nb_games: integer
        """
        
        list_matrices_white_1, list_matrices_white_2, list_matrices_white_3, list_matrices_white_4 = [], [], [], []
        list_matrices_black_1, list_matrices_black_2, list_matrices_black_3, list_matrices_black_4 = [], [], [], []
        list_y_white, list_y_black = [], []
        
        # Get a chunk of data
        df_chunk = self.get_a_chunk(chunk_size)
    
        # Iterate while there are data in self._df_games
        while df_chunk.shape[0] > 0:
            
            # Create game matrices
            matrices_white_1_chunk, matrices_white_2_chunk, matrices_white_3_chunk, matrices_white_4_chunk, \
            matrices_black_1_chunk, matrices_black_2_chunk, matrices_black_3_chunk, matrices_black_4_chunk, \
            y_white_chunk, y_black_chunk = create_game_matrices_one_chunk_games(df_chunk)
            
            # Append to list
            list_matrices_white_1.append(matrices_white_1_chunk)
            list_matrices_white_2.append(matrices_white_2_chunk)
            list_matrices_white_3.append(matrices_white_3_chunk)
            list_matrices_white_4.append(matrices_white_4_chunk)
            list_matrices_black_1.append(matrices_black_1_chunk)
            list_matrices_black_2.append(matrices_black_2_chunk)
            list_matrices_black_3.append(matrices_black_3_chunk)
            list_matrices_black_4.append(matrices_black_4_chunk)
            list_y_white.append(y_white_chunk)
            list_y_black.append(y_black_chunk)
            
            # Update achievement
            with self._lock_nb_games_done:
                self._nb_games_done += df_chunk.shape[0]
                print('Preprocessing: ' + str(self._nb_games_done) + '/' + str(nb_games) + ' done', end='\r')
            
            # Get a chunk of data
            df_chunk = self.get_a_chunk(chunk_size)
            
        # Concat all the data preprocess
        matrices_white_1 = np.concatenate(list_matrices_white_1)
        matrices_white_2 = np.concatenate(list_matrices_white_2)
        matrices_white_3 = np.concatenate(list_matrices_white_3)
        matrices_white_4 = np.concatenate(list_matrices_white_4)
        matrices_black_1 = np.concatenate(list_matrices_black_1)
        matrices_black_2 = np.concatenate(list_matrices_black_2)
        matrices_black_3 = np.concatenate(list_matrices_black_3)
        matrices_black_4 = np.concatenate(list_matrices_black_4)
        y_white = np.concatenate(list_y_white)
        y_black = np.concatenate(list_y_black)
        
        # Use lock to add it to gloabl data for concatenation by principal function create_game_matrices_all_games
        with self._lock_df_games:
            self._list_matrices_white_1.append(matrices_white_1)
            self._list_matrices_white_2.append(matrices_white_2)
            self._list_matrices_white_3.append(matrices_white_3)
            self._list_matrices_white_4.append(matrices_white_4)
            self._list_matrices_black_1.append(matrices_black_1)
            self._list_matrices_black_2.append(matrices_black_2)
            self._list_matrices_black_3.append(matrices_black_3)
            self._list_matrices_black_4.append(matrices_black_4)
            self._list_y_white.append(y_white)
            self._list_y_black.append(y_black)
    
    
    def create_game_matrices_all_games(self, chunk_size=100):
        
        """Return matrices of game matrix for the 4 method for black and white, and an array for is_white_win for white and for black move.
        
        :param chunk_size: size of the chunks of dataframe to preprocess. If memory problems, decrease this number. Default: 100
        :type chunk_size: integer
        
        :return matrices_white_1: matrix of game matrices (1 matrix for one position in the game) for white move using first method
        :rtype matrices_white_1: np.array of shape (x, 8, 8, 12) (where x is the number of white position of the game), dtype=bool

        :return matrices_black_1: matrix of game matrices (1 matrix for one position in the game) for black move using first method
        :rtype matrices_black_1: np.array of shape (x, 8, 8, 12) (where x is the number of black position of the game), dtype=bool

        :return matrices_white_2: matrix of game matrices (1 matrix for one position in the game) for white move using second method
        :rtype matrices_white_2: np.array of shape (x, 8, 8, 6) (where x is the number of white position of the game), dtype=int

        :return matrices_black_2: matrix of game matrices (1 matrix for one position in the game) for black move using second method
        :rtype matrices_black_2: np.array of shape (x, 8, 8, 6) (where x is the number of black position of the game), dtype=int

        :return matrices_white_3: matrix of game matrices (1 matrix for one position in the game) for white move using third method
        :rtype matrices_white_3: np.array of shape (x, 8, 8, 4) (where x is the number of white position of the game), dtype=float

        :return matrices_black_3: matrix of game matrices (1 matrix for one position in the game) for black move using third method
        :rtype matrices_black_3: np.array of shape (x, 8, 8, 4) (where x is the number of black position of the game), dtype=float

        :return matrices_white_4: matrix of game matrices (1 matrix for one position in the game) for white move using fourth method
        :rtype matrices_white_4: np.array of shape (x, 8, 8, 2) (where x is the number of white position of the game), dtype=float

        :return matrices_black_4: matrix of game matrices (1 matrix for one position in the game) for black move using fourth method
        :rtype matrices_black_4: np.array of shape (x, 8, 8, 2) (where x is the number of black position of the game), dtype=float

        :return y_white: array with full of True or False depending of is_white_win
        :rtype y_white: np.array of shape (x) dtype=bool

        :return y_black: array with full of True or False depending of is_white_win
        :rtype y_black: np.array of shape (x) dtype=bool
        """
        
        threads = []
        n_core = multiprocessing.cpu_count()
        nb_games = self._df_games.shape[0]
        
        # Create and start thread to do the task
        for n_thread in range(0, n_core):
            threads.append(threading.Thread(target=self.thread_create_game_matrices_all_game, args=(chunk_size, nb_games, )))
            threads[n_thread].start()
            
        for thread in threads:
            thread.join()
            
        # Concat data from threads
        matrices_white_1 = np.concatenate(self._list_matrices_white_1)
        matrices_white_2 = np.concatenate(self._list_matrices_white_2)
        matrices_white_3 = np.concatenate(self._list_matrices_white_3)
        matrices_white_4 = np.concatenate(self._list_matrices_white_4)
        matrices_black_1 = np.concatenate(self._list_matrices_black_1)
        matrices_black_2 = np.concatenate(self._list_matrices_black_2)
        matrices_black_3 = np.concatenate(self._list_matrices_black_3)
        matrices_black_4 = np.concatenate(self._list_matrices_black_4)
        y_white = np.concatenate(self._list_y_white)
        y_black = np.concatenate(self._list_y_black)
        
        return matrices_white_1, matrices_black_1, matrices_white_2, matrices_black_2, \
               matrices_white_3, matrices_black_3, matrices_white_4, matrices_black_4, \
               y_white, y_black
        
        
    @property
    def _df_games(self):
        return self.__df_games

    
    @_df_games.setter
    def _df_games(self, df_games):
        self.__df_games = df_games