import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

class ChessDatasetTuning(Dataset):
    
    def __init__(self, color_dataset, path_data='./data/', nb_splits_CV=2, random_state=42, memory_map=True):
        
        self._memory_map = memory_map
        self._nb_splits_CV = nb_splits_CV
        self._random_state = random_state
        self._mode = 'training'
        self._train_index = None
        self._test_index = None
        self.read_data(color_dataset=color_dataset, path_data=path_data)
        self.init_kf_CV_iter()
        
        
    def __len__(self):
        
        if self._mode == 'training': 
            len_dataset = self._train_index.shape[0]
        elif self._mode == 'testing': 
            len_dataset = self._test_index.shape[0]
            
        return len_dataset
        
        
    def __getitem__(self, idx):
        
        
        if self._mode == 'training': 
            train_idx = self._train_index[idx]
            if self._memory_map == True:
                item = {'X_train': np.array(self._X[train_idx]), 'y_train': np.array(self._y[train_idx])}
            else:
                item = {'X_train': self._X[train_idx], 'y_train': self._y[train_idx]}
        elif self._mode == 'testing': 
            test_idx = self._test_index[idx]
            if self._memory_map == True:
                item = {'X_test': np.array(self._X[test_idx]), 'y_test': np.array(self._y[test_idx])}
            else:
                item = {'X_test': self._X[test_idx], 'y_test': self._y[test_idx]}
            
        return item
    
    
    def read_data(self, color_dataset, path_data):
                
        self._X = np.memmap(path_data + 'X_' + color_dataset + '_tuning.dat', dtype=bool, mode='r')
        self._X = self._X.reshape((int(self._X.shape[0] / (12 * 8 * 8)), 12, 8, 8))
        self._y = np.memmap(path_data + 'y_' + color_dataset + '_tuning.dat', dtype=bool, mode='r')
        
        if self._memory_map == False:
    
            self._X = torch.Tensor(np.array(self._X)).float()
            self._y = torch.Tensor(np.array(self._y)).float()
            
            mean = self._X.mean()
            std = self._X.std()
            
            for channel in range(0, 12):
                self._X[:, channel] = (self._X[:, channel] - mean) / std
            
            
    def init_kf_CV_iter(self):
        
        self._kf_CV_iter = iter(KFold(n_splits=self._nb_splits_CV, random_state=self._random_state, shuffle=True).split(self._X))
    
    
    def update_set_CV(self):
        
        index = next(self._kf_CV_iter)
        self._train_index = index[0]
        self._test_index = index[1] 
        
        
    def set_mode(self, mode):
        
        self._mode = mode