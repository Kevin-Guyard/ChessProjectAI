import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

class ChessDatasetTuning(Dataset):
    
    def __init__(self, color_dataset, n_method, shape_X, path_data='./data/', nb_splits_CV=2, random_state=42):
        
        path_X, path_y, dtype_X = self.get_path_and_dtype(path_data, color_dataset, n_method)
        
        self._X = np.memmap(path_X, dtype=dtype_X, mode='r')
        self._X = self._X.reshape((int(self._X.shape[0] / np.prod(shape_X)), ) + shape_X)
        self._y = np.memmap(path_y, dtype=bool, mode='r')
        self._kf_CV_iter = iter(KFold(n_splits=nb_splits_CV, random_state=random_state, shuffle=True).split(self._X))
        self._mode = 'training'
        self._train_index = None
        self._test_index = None
        
        
    def __len__(self):
        
        if self._mode == 'training': 
            len_dataset = self._train_index.shape[0]
        elif self._mode == 'testing': 
            len_dataset = self._test_index.shape[0]
            
        return len_dataset
        
        
    def __getitem__(self, idx):
        
        
        if self._mode == 'training': 
            train_idx = self._train_index[idx]
            item = {'X_train': np.array(self._X[train_idx]), 'y_train': np.array(self._y[train_idx])}
        elif self._mode == 'testing': 
            test_idx = self._test_index[idx]
            item = {'X_test': np.array(self._X[test_idx]), 'y_test': np.array(self._y[test_idx])}
            
        return item
    
    
    def update_set_CV(self):
        
        index = next(self._kf_CV_iter)
        self._train_index = index[0]
        self._test_index = index[1] 
        
        
    def set_mode(self, mode):
        
        self._mode = mode
        
        
    def get_path_and_dtype(self, path_data, color_dataset, n_method):
        
        path_X = path_data + 'X_' + color_dataset + '_' + str(n_method) + '_tuning.dat'
        path_y = path_data + 'y_' + color_dataset + '_tuning.dat'
        
        if n_method == 1: 
            dtype_X = bool
        elif n_method == 2: 
            dtype_X = int
        elif n_method == 3: 
            dtype_X = float
        elif n_method == 4: 
            dtype_X = float
            
        return path_X, path_y, dtype_X