
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def plot_rnn(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses)), losses, label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('RNN Regression Model Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('./evaluations/rnn_loss.png')
    plt.show()

def plot_elbow(train_errors, test_errors):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 15), train_errors, label='Train MSE')
    plt.plot(range(1, 15), test_errors, label='Test MSE')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('MSE')
    plt.title('KNN Model Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('./evaluations/knn_elbow.png')
    plt.show()

def plot_cnn_rnn(epochs, cnn_loss, rnn_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1)  , cnn_loss, label='CNN Training Loss', marker='o')
    plt.plot(range(1, epochs + 1)  , rnn_loss, label='RNN Training Loss', marker='x')
    plt.title('Comparison of Training Losses: CNN vs RNN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./evaluations/cnn_rnn_loss.png')
    plt.show()

def plot_knn_rfb_linear(x, y, model, model2, model3):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_pred_train1 = model.predict(X_train_scaled)
    y_pred_test1 = model.predict(X_test_scaled)

    y_pred_train2 = model2.predict(X_train_scaled)
    y_pred_test2 = model2.predict(X_test_scaled)

    y_pred_train3 = model3.predict(X_train_scaled)
    y_pred_test3 = model3.predict(X_test_scaled)

    # MSE
    mse_train1 = mean_squared_error(y_train, y_pred_train1)
    mse_test1 = mean_squared_error(y_test, y_pred_test1)

    mse_train2 = mean_squared_error(y_train, y_pred_train2)
    mse_test2 = mean_squared_error(y_test, y_pred_test2)

    mse_train3 = mean_squared_error(y_train, y_pred_train3)
    mse_test3 = mean_squared_error(y_test, y_pred_test3)
    
    # Plot RMSE comparison for all three models
    labels = ['KNN Regresssion', 'Random Forrest', 'Linear Regression']
    train_rmse_values = [mse_train1, mse_train2, mse_train3]
    test_rmse_values = [mse_test1, mse_test2, mse_test3]

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(labels)) - 0.2, train_rmse_values, width=0.4, label='Train MSE')
    plt.bar(np.arange(len(labels)) + 0.2, test_rmse_values, width=0.4, label='Test MSE')

    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel('Mean Squared Error')
    plt.title('MSE for three Models')
    plt.legend()
    plt.savefig('./evaluations/knn_rfb_lrg_mse.png')
    plt.show()

    # Calculate RMSE for all three models
    rmse_train1 = np.sqrt(mean_squared_error(y_train, y_pred_train1))
    rmse_test1 = np.sqrt(mean_squared_error(y_test, y_pred_test1))

    rmse_train2 = np.sqrt(mean_squared_error(y_train, y_pred_train2))
    rmse_test2 = np.sqrt(mean_squared_error(y_test, y_pred_test2))

    rmse_train3 = np.sqrt(mean_squared_error(y_train, y_pred_train3))
    rmse_test3 = np.sqrt(mean_squared_error(y_test, y_pred_test3))

    # Plot RMSE comparison for all three models
    train_rmse_values = [rmse_train1, rmse_train2, rmse_train3]
    test_rmse_values = [rmse_test1, rmse_test2, rmse_test3]

    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(len(labels)) - 0.2, train_rmse_values, width=0.4, label='Train RMSE')
    plt.bar(np.arange(len(labels)) + 0.2, test_rmse_values, width=0.4, label='Test RMSE')

    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel('Root Mean Squared Error')
    plt.title('RMSE Comparison for Three Models')
    plt.legend()
    plt.savefig('./evaluations/knn_rfb_lrg_rmse.png')
    plt.show()

    # Calculate R2 score for all three models
    r2_train1 = r2_score(y_train, y_pred_train1)
    r2_test1 = r2_score(y_test, y_pred_test1)

    r2_train2 = r2_score(y_train, y_pred_train2)
    r2_test2 = r2_score(y_test, y_pred_test2)

    r2_train3 = r2_score(y_train, y_pred_train3)
    r2_test3 = r2_score(y_test, y_pred_test3)

    # Plot R2 score comparison for all three models
    train_r2_values = [r2_train1, r2_train2, r2_train3]
    test_r2_values = [r2_test1, r2_test2, r2_test3]

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(labels)) - 0.2, train_r2_values, width=0.4, label='Train R2 Score')
    plt.bar(np.arange(len(labels)) + 0.2, test_r2_values, width=0.4, label='Test R2 Score')

    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel('R2 Score')
    plt.title('R2 Score Comparison for Three Models')
    plt.legend()
    plt.savefig('./evaluations/knn_rfb_lrg_r2.png')
    plt.show()

def plot_mse(y_pred_train, y_pred_test, y_train, y_test, algo):

    # MSE
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Plot MSE comparison
    labels = ['Train MSE', 'Test MSE']
    mse_values = [mse_train, mse_test]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, mse_values, color=['blue', 'orange'])
    plt.ylabel('Mean Squared Error')
    plt.title('Model Evaluation - Train vs Test MSE')
    plt.savefig('./evaluations/' + str(algo) + '_mse.png')
    plt.show()

def plot_rmse(y_pred_train, y_pred_test, y_train, y_test, algo):

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)


    # Plot RMSE comparison
    labels = ['Train RMSE', 'Test RMSE']
    rmse_values = [rmse_train, rmse_test]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, rmse_values, color=['blue', 'orange'])
    plt.ylabel('Root Mean Squared Error')
    plt.title('Model Evaluation - Train vs Test RMSE')
    plt.savefig('./evaluations/' + str(algo) + '_rmse.png')
    plt.show()

def plot_r2(y_pred_train, y_pred_test, y_train, y_test, algo):

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    labels = ['Train R2 Score', 'Test R2 Score']
    r2_values = [r2_train, r2_test]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, r2_values, color=['blue', 'orange'])
    plt.ylabel('R2 Score')
    plt.title('Model Evaluation - Train vs Test R2 Score')
    plt.savefig('./evaluations/' + str(algo) + '_r2.png')
    plt.show()