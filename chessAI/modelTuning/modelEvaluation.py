import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from chessAI.datasets import ChessDatasetTuning
from chessAI.models import get_model

def evaluate_model_accuracy_CV(color_dataset, n_method, parameters, path_data='./data/', path_temp='./temp/', n_epochs=100, batch_size=100, nb_splits_CV=2, tolerance=1e-7, random_state=42):
    
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    
    accuracies_test = []
    dataset = ChessDatasetTuning(color_dataset=color_dataset, n_method=n_method, shape_X=parameters['shape_X'], path_data=path_data, nb_splits_CV=nb_splits_CV, random_state=random_state)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    # CV loop
    while True:
            
        try:
            dataset.update_set_CV()
        except StopIteration:
            break
            
        torch.manual_seed(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
            
        model = get_model(parameters=parameters)
        
        if torch.cuda.is_available():
            model = model.to(device='cuda')
            
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])
        
        losses_train = []
        
        for epoch in range(0, n_epochs):
            
            losses_train_batch = []
            dataset.set_mode('training')
            iterator = iter(dataloader)
            model.train()
            
            # Training loop
            while True:
                
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                    
                X_train, y_train = batch['X_train'].float(), batch['y_train'].float()
                
                if torch.cuda.is_available():
                    X_train, y_train = X_train.to(device='cuda'), y_train.to(device='cuda')
                    
                optimizer.zero_grad()
                outputs_training = torch.squeeze(model(X_train))
                loss = criterion(outputs_training, y_train)
                            
                loss.backward()
                optimizer.step()
            
                losses_train_batch.append(loss.item())
                
            losses_train.append(np.mean(losses_train_batch))
            
            if epoch > 0 and np.abs(losses_train[-1] - losses_train[-2]) < tolerance:
                break
            
        nb_correct_pred = 0
        nb_total_pred = 0
        dataset.set_mode('testing')
        iterator = iter(dataloader)
        model.eval()
        
        # Testing loop
        while True:
            
            try:
                batch = next(iterator)
            except StopIteration:
                break
                
            X_test, y_test = batch['X_test'].float(), batch['y_test'].float()
            
            if torch.cuda.is_available():
                X_test = X_test.to(device='cuda')
                
            with torch.no_grad():
            
                y_pred = torch.squeeze(model(X_test))
            
                if torch.cuda.is_available():
                    y_pred = y_pred.to(device='cpu')

                y_pred = y_pred.round().detach().numpy()
                nb_total_pred += y_test.shape[0]
                nb_correct_pred += np.sum(y_pred == y_test.detach().numpy())
            
        accuracy_test = 100 * nb_correct_pred / nb_total_pred
        accuracies_test.append(accuracy_test)
        
    accuracy_test_CV = np.mean(accuracies_test)
    
    return accuracy_test_CV