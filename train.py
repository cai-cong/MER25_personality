# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from math import sqrt
import os



def train_model(model, trainloader, devloader, epochs, lr, log_file_name):
    
    model_path = os.path.join("./model/",log_file_name)
    os.makedirs(model_path, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_rmse = [float('inf')] * 5  
    best_epoch = [0] * 5  
    

    print('Start training')
    for epoch in range(1, epochs + 1):

        train_loss = train(model, trainloader, optimizer)
        val_metrics = evaluate(model, devloader)

        print('-' * 50)
        print(f'Epoch:{epoch:>3} | [Train] | Loss: {train_loss:.3f}')
        for dim in range(5):
            print(f'Epoch:{epoch:>3} | [Val][Dim {dim + 1}] | RMSE: {val_metrics["rmse"][dim]:.3f}  | PCC: {val_metrics["pcc"][dim]:.3f} | CCC: {val_metrics["ccc"][dim]:.3f}')
        print('-' * 50)

        torch.save(model, os.path.join(model_path, f"{epoch}.pth"))

        # Update best result and eopch
        for dim in range(5):
            if val_metrics["rmse"][dim] < best_val_rmse[dim]:
                best_val_rmse[dim] = val_metrics["rmse"][dim]
                best_epoch[dim] = epoch

    print(f"Best val RMSE per dimension: {best_val_rmse}")
    print(f"Test RMSE for best val RMSE: {best_epoch}")

def train(model, trainloader, optimizer):
    model.train()
    running_loss = 0.0
    criterion = torch.nn.MSELoss()  

    for train_data in trainloader:
        features, labels, lengths = train_data

        features, labels = features.cuda(), labels.cuda()

        optimizer.zero_grad()
        predictions = model(features)  #  (batch_size, 5)

        loss = criterion(predictions, labels.float())  
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)



def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        predictions_corr = np.empty((0, 5))  
        labels_corr = np.empty((0, 5))      

        for data in dataloader:
            features, labels, lengths = data
            predictions = model(features.cuda())  

            # Check if the output dimension of the model is (batch_size, 5)
            assert predictions.shape[1] == 5, f"Expected predictions to have shape (batch_size, 5), but got {predictions.shape}"

            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            predictions_corr = np.append(predictions_corr, predictions, axis=0)
            labels_corr = np.append(labels_corr, labels, axis=0)

        # Calculate the evaluation indicators for each personality dimension separately
        rmse_list = []
        pcc_list = []
        ccc_list = []
        for dim in range(5):

            rmse = sqrt(mean_squared_error(predictions_corr[:, dim], labels_corr[:, dim]))
            rmse_list.append(rmse)

            # PCC (Pearson Correlation Coefficient)
            pcc = np.corrcoef(predictions_corr[:, dim], labels_corr[:, dim])[0, 1]
            pcc_list.append(pcc)

            # CCC (Concordance Correlation Coefficient)
            ccc = ccc_score(predictions_corr[:, dim], labels_corr[:, dim])
            ccc_list.append(ccc)

        return {
            "rmse": rmse_list,
            "pcc": pcc_list,
            "ccc": ccc_list,
        }


# 计算CCC（Concordance Correlation Coefficient）
def ccc_score(predictions, labels):
    mean_pred = np.mean(predictions)
    mean_true = np.mean(labels)

    var_pred = np.var(predictions)
    var_true = np.var(labels)

    covariance = np.mean((predictions - mean_pred) * (labels - mean_true))

    ccc = (2 * covariance) / (var_pred + var_true + (mean_pred - mean_true) ** 2)
    return ccc
