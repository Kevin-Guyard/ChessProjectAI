import pandas as pd
import numpy as np
import threading
import multiprocessing
from chessAI.preprocessing.gameMatricesCreation import create_game_matrices_one_chunk_games

class Preprocesser():
    
    def __init__(self, df_games):
        
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
        
        with self._lock_df_games:
            if self._df_games.shape[0] == 0: 
                df_chunk = pd.DataFrame()
            else:
                if self._df_games.shape[0] > chunk_size:
                    df_chunk = self._df_games[self._df_games.shape[0] - chunk_size:self._df_games.shape[0]].copy()
                    self._df_games.drop(index=[x for x in range(self._df_games.shape[0] - chunk_size, self._df_games.shape[0])], inplace=True)
                else:
                    df_chunk = self._df_games[0:self._df_games.shape[0]].copy()
                    self._df_games.drop(index=[x for x in range(0, self._df_games.shape[0])], inplace=True)
                    
        return df_chunk
        
        
    def thread_create_game_matrices_all_game(self, chunk_size, nb_games):
        
        list_matrices_white_1, list_matrices_white_2, list_matrices_white_3, list_matrices_white_4 = [], [], [], []
        list_matrices_black_1, list_matrices_black_2, list_matrices_black_3, list_matrices_black_4 = [], [], [], []
        list_y_white, list_y_black = [], []
        
        df_chunk = self.get_a_chunk(chunk_size)
    
        while df_chunk.shape[0] > 0:
            
            matrices_white_1_chunk, matrices_white_2_chunk, matrices_white_3_chunk, matrices_white_4_chunk, \
            matrices_black_1_chunk, matrices_black_2_chunk, matrices_black_3_chunk, matrices_black_4_chunk, \
            y_white_chunk, y_black_chunk = create_game_matrices_one_chunk_games(df_chunk)
            
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
            
            with self._lock_nb_games_done:
                self._nb_games_done += df_chunk.shape[0]
                print('Preprocessing: ' + str(self._nb_games_done) + '/' + str(nb_games) + ' done', end='\r')
            
            df_chunk = self.get_a_chunk(chunk_size)
            
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
        
        threads = []
        nb_cpu = multiprocessing.cpu_count()
        nb_games = self._df_games.shape[0]
        
        # Create and start thread to do the task
        for n_thread in range(0, nb_cpu):
            
            threads.append(threading.Thread(target=self.thread_create_game_matrices_all_game, args=(chunk_size, nb_games, )))
            threads[n_thread].start()
            
        for thread in threads:
            thread.join()
            
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