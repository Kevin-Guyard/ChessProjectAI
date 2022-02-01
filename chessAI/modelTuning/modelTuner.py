import pandas as pd
import os
import json
from chessAI.modelHyperParameters import get_parameters_tuning
from chessAI.modelTuning.modelEvaluation import evaluate_model_accuracy_CV

class ModelTuner():
    
    def __init__(self):
        
        self._n_chunk = 1
        
        
    def tuning(self, color_dataset, n_method, path_data='./data/', path_temp='./temp/', n_epochs=100, batch_size=100, nb_splits_CV=2, tolerance=1e-7, random_state=42, memory_map=True):
        
        if not os.path.exists(path_temp + 'tuning_data/'):
            os.mkdir(path_temp + 'tuning_data/')
            
        if not os.path.exists(path_temp + 'backup/'):
            os.mkdir(path_temp + 'backup/')
        
        if not os.path.exists(path_temp + 'backup/tuning_backup.json'):
            n_chunk = 1
        else:
            with open(path_temp + 'backup/tuning_backup.json', 'r') as file:
                backup = json.load(file)
                n_chunk = backup['n_chunk']
                
        parameters_tuning = get_parameters_tuning(n_method)
        
        while n_chunk != self._n_chunk:
            parameters_tuning.pop(0)
            self._n_chunk += 1
        
        for parameters in parameters_tuning:
            
            accuracy_test_CV = evaluate_model_accuracy_CV(color_dataset=color_dataset, n_method=n_method, parameters=parameters, path_data=path_data, path_temp=path_temp, n_epochs=n_epochs, batch_size=batch_size, nb_splits_CV=nb_splits_CV, tolerance=tolerance, random_state=random_state, memory_map=memory_map)
            
            dic_result = {'accuracy_test_CV': accuracy_test_CV}
            dic_result.update(parameters)
            df_result = pd.DataFrame(dic_result)
            df_result.to_csv(path_temp + 'tuning_data/df_result_' + str(self._n_chunk) + '.csv.gz', encoding='utf-8', compression='gzip', sep='\t', index=False)
            
            self._n_chunk += 1
            with open(path_temp + 'backup/tuning_backup.json', 'w') as file:
                json.dump({'n_chunk': self._n_chunk}, file)
            
        files = os.listdir(path_temp + 'tuning_data/')
        
        list_df_result = []
        
        for file in files:
            if 'df_result' in file:
                df_result = pd.read_csv(path_temp + 'tuning_data/' + file, compression='gzip', encoding='utf-8', sep='\t')
                list_df_result.append(df_result)
                
        for file in files:
            if 'df_result' in file:
                os.remove(path_temp + 'tuning_data/' + file)
                
        os.rmdir(path_temp + 'tuning_data/')
        os.remove(path_temp + 'backup/tuning_backup.json')
        os.rmdir(path_temp + 'backup/')
                
        df_tuning = pd.concat([df for df in list_df_result]).reset_index(drop=True)
                    
        return df_tuning