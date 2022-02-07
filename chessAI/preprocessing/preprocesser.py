import pandas as pd
import numpy as np
import threading
import multiprocessing
import os
from sklearn.model_selection import train_test_split
from chessAI.preprocessing.gameMatricesCreation import create_game_matrices_one_chunk_games

class Preprocesser():
    
    def __init__(self, df_games, nb_white_moves_done=0, nb_black_moves_done=0):
        
        self._df_games = df_games
        self._nb_games_done = 0
        self._nb_white_moves_done = nb_white_moves_done
        self._nb_black_moves_done = nb_black_moves_done
        self._n_chunk = 1
        self._lock_df_games = threading.Lock()
        self._lock_nb_games_done = threading.Lock()
        self._lock_nb_moves_done = threading.Lock()
        self._lock_n_chunk = threading.Lock()     
        
        
    def get_a_chunk(self, chunk_size):
        
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
        
        
    def thread_create_game_matrices_chunks(self, chunk_size, nb_games, path_temp):

        nb_moves_done = 0
        
        # Get a chunk of data
        df_chunk = self.get_a_chunk(chunk_size)
        
        # Iterate while there are data in self._df_games
        while df_chunk.shape[0] > 0:
            
            # Create game matrices
            matrices_white_chunk, matrices_black_chunk, y_white_chunk, y_black_chunk = create_game_matrices_one_chunk_games(df_chunk)

            # Get the number of the actual chunk and update (+1)
            with self._lock_n_chunk:
                n_chunk = self._n_chunk
                self._n_chunk += 1
            
            # Save chunk in path temp
            np.savez_compressed(path_temp + 'X_white/' + 'X_white_chunk_' + str(n_chunk), matrices=matrices_white_chunk)
            np.savez_compressed(path_temp + 'X_black/' + 'X_black_chunk_' + str(n_chunk), matrices=matrices_black_chunk)
            np.savez_compressed(path_temp + 'y_white/' + 'y_white_chunk_' + str(n_chunk), matrices=y_white_chunk)
            np.savez_compressed(path_temp + 'y_black/' + 'y_black_chunk_' + str(n_chunk), matrices=y_black_chunk)
            
            # Update nb of white and black move total
            with self._lock_nb_moves_done:
                self._nb_white_moves_done += y_white_chunk.shape[0]
                self._nb_black_moves_done += y_black_chunk.shape[0]
            
            # Update achievement
            with self._lock_nb_games_done:
                self._nb_games_done += df_chunk.shape[0]
                print('Preprocessing: ' + str(self._nb_games_done) + '/' + str(nb_games) + ' done', end='\r')
                
            # Get a chunk of data for next iteration
            df_chunk = self.get_a_chunk(chunk_size)

            
    def thread_unify_chunk(self, target, path_temp):
        
        # Choose the shape of the matrice and the dtype according to the target
        if target == 'X_white':
            shape_matrice = (self._nb_white_moves_done, 12, 8, 8)
        elif target == 'X_black':
            shape_matrice = (self._nb_black_moves_done, 12, 8, 8)
        elif target == 'y_white':
            shape_matrice = (self._nb_white_moves_done,)
        elif target == 'y_black':
            shape_matrice = (self._nb_black_moves_done,)
                    
        # Read the name of the files (which are the chunk for the given target)
        files = os.listdir(path_temp + target + '/')
        # Create mapped matrices
        matrices = np.memmap(path_temp + target + '.dat', dtype=bool, mode='w+', shape=shape_matrice)
        
        index_start = 0
        
        # Iterate over chunks: load the chunk and add it to the mapped matrices
        for file in files:
            matrices_chunk = np.load(path_temp + target + '/' + file)['matrices']
            index_stop = index_start + matrices_chunk.shape[0]
            matrices[index_start:index_stop] = matrices_chunk
            index_start = index_stop
            
        # Delete from RAM
        del matrices
            
        # Remove chunks
        for file in files:
            os.remove(path_temp + target + '/' + file)
            
        # Remove the directory
        os.rmdir(path_temp + target + '/')
        
        
    def split_dataset(self, path_temp='./temp/', path_data='./data/', size_validation=0.2, random_state=42):
        
        targets = ['X_white', 'X_black', 'y_white', 'y_black']
        
        # Initialize the random seed to reproductibility
        np.random.seed(random_state)

        # Create index from 0 to last move
        index_white = np.arange(0, self._nb_white_moves_done)
        index_black = np.arange(0, self._nb_black_moves_done)
        
        # Shuffle the index
        np.random.shuffle(index_white)
        np.random.shuffle(index_black)
        
        # Compute the number of moves in validation set
        nb_val_white = int(size_validation * self._nb_white_moves_done)
        nb_val_black = int(size_validation * self._nb_black_moves_done)
        
        # Split the index between validation and training
        val_index_white = index_white[:nb_val_white]
        train_index_white = index_white[nb_val_white:]
        val_index_black = index_black[:nb_val_black]
        train_index_black = index_black[nb_val_black:]
        
        # Iterate over targets
        for target in targets:
            # Choose parameters regarding the target
            if target == 'X_white':
                shape_matrice = (self._nb_white_moves_done, 12, 8, 8)
                shape_matrice_val = (nb_val_white, 12, 8, 8)
                shape_matrice_train = (self._nb_white_moves_done - nb_val_white, 12, 8, 8)
                val_index = val_index_white
                train_index = train_index_white
            elif target == 'y_white':
                shape_matrice = (self._nb_white_moves_done, )
                shape_matrice_val = (nb_val_white, )
                shape_matrice_train = (self._nb_white_moves_done - nb_val_white, )
                val_index = val_index_white
                train_index = train_index_white
            elif target == 'X_black':
                shape_matrice = (self._nb_black_moves_done, 12, 8, 8)
                shape_matrice_val = (nb_val_black, 12, 8, 8)
                shape_matrice_train = (self._nb_black_moves_done - nb_val_black, 12, 8, 8)
                val_index = val_index_black
                train_index = train_index_black
            elif target == 'y_black':
                shape_matrice = (self._nb_black_moves_done, )
                shape_matrice_val = (nb_val_black, )
                shape_matrice_train = (self._nb_black_moves_done - nb_val_black, )
                val_index = val_index_black
                train_index = train_index_black
                            
            # Read the matrices
            matrices = np.memmap(path_temp + target + '.dat', dtype=bool, mode='r', shape=shape_matrice)
            # Create new matrices for validation and training set
            matrices_val = np.memmap(path_data + target + '_val.dat', dtype=bool, mode='w+', shape=shape_matrice_val)
            matrices_train = np.memmap(path_data + target + '_tuning.dat', dtype=bool, mode='w+', shape=shape_matrice_train)
            
            # Assign value to matrices
            matrices_val[:] = matrices[val_index]
            matrices_train[:] = matrices[train_index]
            
            # Delete from RAM
            del matrices
            del matrices_val
            del matrices_train
            
            # Delete from disk
            os.remove(path_temp + target + '.dat')
            
            print('Preprocessing: ' + target + ' split done               ', end='\r')
            
        print('Preprocessing: all splits done                      ', end='\r')
            
            
    def create_game_matrices_chunks(self, chunk_size=100, path_temp='./temp/'):
        
        threads_chunk = []
        n_core = multiprocessing.cpu_count()
        targets = ['X_white', 'X_black', 'y_white', 'y_black']
        nb_games = self._df_games.shape[0]
        
        # Reset index (usefull if the given df is a fragment of a df)
        self._df_games.reset_index(drop=True, inplace=True)
        
        # Create temp subdirectory
        for target in targets:
            if not os.path.exists(path_temp + target + '/'):
                os.makedirs(path_temp + target + '/')
        
        # Create and start thread to do the task of martices creation
        for n_thread in range(0, n_core):
            threads_chunk.append(threading.Thread(target=self.thread_create_game_matrices_chunks, args=(chunk_size, nb_games, path_temp, )))
            threads_chunk[n_thread].start()
            
        # Wait the end of workers
        for thread in threads_chunk:
            thread.join()
            
            
    def unify_chunk(self, path_temp='./temp/'):
        
        targets = ['X_white', 'X_black', 'y_white', 'y_black']
        threads_unify = []
        
        # Create and start thread to do the task of unify matrices
        for target in targets:
            thread_unify = threading.Thread(target=self.thread_unify_chunk, args=(target, path_temp, ))
            thread_unify.start()
            threads_unify.append(thread_unify)
            
        # Wait the end of workers
        for thread_unify in threads_unify:
            thread_unify.join()
            
        print('Preprocessing: unification done                 ', end='\r')