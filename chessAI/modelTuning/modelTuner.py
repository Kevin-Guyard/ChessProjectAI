import pandas as pd
from chessAI.modelHyperParameters import get_parameters_tuning
from chessAI.modelTuning.modelEvaluation import evaluate_model_accuracy_CV

class ModelTuner():
    
    def __init__(self):
        
        True #TODO
        
        
    def tuning(self, color_dataset, n_method, path_data='./data/', n_epochs=100, batch_size=100, n_splits_CV=2, tolerance=1e-7, random_state=42):
        
        list_df_result = []
        
        parameters_tuning = get_parameters_tuning(n_method)
        
        for parameters in parameters_tuning:
            
            accuracy_test_CV = evaluate_model_accuracy_CV(color_dataset=color_dataset, n_method=n_method, parameters=parameters, path_data=path_data, n_epochs=n_epochs, batch_size=batch_size, n_splits_CV=n_splits_CV, tolerance=tolerance, random_state=random_state)
            
            dic_result = {'accuracy_test_CV': accuracy_test_CV}.update(parameters)
            df_result = pd.DataFrame(dic_result)
            
        df_tuning = pd.concat([df for df in list_df_result]).reset_index(drop=True)
        
        return df_tuning