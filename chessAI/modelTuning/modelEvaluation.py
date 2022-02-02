import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from chessAI.datasets import ChessDatasetTuning
from chessAI.models import get_model

def save_model(model_state_dict, optimizer_state_dict, losses_train, accuracies_test, epoch, n_split_CV, is_training, path_temp='./temp/'):
    
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'losses_train': losses_train,
        'accuracies_test': accuracies_test,
        'epoch': epoch,
        'n_split_CV': n_split_CV,
        'is_training': is_training
    }, path_temp + 'backup/model_backup.pth')
    
    
def load_model(path_temp):
    
    if not os.path.exists(path_temp + 'backup/model_backup.pth'):
        checkpoint = None
    else:
        checkpoint = torch.load(path_temp + 'backup/model_backup.pth')
        
    return checkpoint
    

def evaluate_model_accuracy_CV(color_dataset, n_method, parameters, path_data='./data/', path_temp='./temp/', n_epochs=100, batch_size=100, nb_splits_CV=2, tolerance=1e-7, random_state=42, memory_map=True):
    
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    
    accuracies_test = []
    dataset = ChessDatasetTuning(color_dataset=color_dataset, n_method=n_method, path_data=path_data, nb_splits_CV=nb_splits_CV, random_state=random_state, memory_map=memory_map)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    n_split_CV = 0
    
    checkpoint = load_model(path_temp=path_temp)
    
    if checkpoint != None:
        accuracies_test = checkpoint['accuracies_test'].copy()
        for n in range(1, checkpoint['n_split_CV']):
            dataset.update_set_CV()
            n_split_CV += 1
        
    # CV loop
    while True:
        
        losses_train = []
 
        try:
            dataset.update_set_CV()
            n_split_CV +=1
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
        
        if checkpoint != None and checkpoint['is_training'] == True:
            losses_train = checkpoint['losses_train'].copy()
            n_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint = None
        else:
            losses_train = []
            n_epoch = 0
        
        for epoch in range(n_epoch, n_epochs):
            
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
                    X_train = X_train.to(device='cuda')
                    y_train = y_train.to(device='cuda')
                    
                optimizer.zero_grad()
                outputs_training = torch.squeeze(model(X_train), 1)
                loss = criterion(outputs_training, y_train)
                            
                loss.backward()
                optimizer.step()
            
                losses_train_batch.append(loss.item())
                
            losses_train.append(np.mean(losses_train_batch))
            
            if epoch > 0 and np.abs(losses_train[-1] - losses_train[-2]) < tolerance:
                save_model(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), losses_train=losses_train, \
                           accuracies_test=accuracies_test, epoch=n_epochs, n_split_CV=n_split_CV, is_training=True, path_temp=path_temp)
                break
            else:
                save_model(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), losses_train=losses_train, \
                           accuracies_test=accuracies_test, epoch=epoch+1, n_split_CV=n_split_CV, is_training=True, path_temp=path_temp)
            
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
                
            X_test, y_test = batch['X_test'].float(), batch['y_test']
            
            if torch.cuda.is_available():
                X_test = X_test.to(device='cuda')
                
            with torch.no_grad():
            
                y_pred = torch.squeeze(model(X_test))
            
                if torch.cuda.is_available():
                    y_pred = y_pred.to(device='cpu')
                    y_test = y_test.to(device='cpu')

                y_pred = y_pred.round().detach().numpy()
                nb_total_pred += y_test.shape[0]
                nb_correct_pred += np.sum(y_pred == y_test.detach().numpy())
            
        accuracy_test = 100 * nb_correct_pred / nb_total_pred
        accuracies_test.append(accuracy_test)
        
        save_model(model_state_dict=None, optimizer_state_dict=None, losses_train=None, accuracies_test=accuracies_test, \
                   epoch=None, n_split_CV=n_split_CV+1, is_training=False, path_temp=path_temp)
                
    accuracy_test_CV = np.mean(accuracies_test)
    
    os.remove(path_temp + 'backup/model_backup.pth')
    
    return accuracy_test_CV