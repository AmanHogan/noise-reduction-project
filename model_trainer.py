import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


class ModelTrainer():

    def __init__(self, x, y, algorithm):
        self.x = x
        self.y = y
        self.algorithm = algorithm

    def run_trainer(self):
        model = None
        if self.algorithm == 'knn':
            model = self.train_knn()

        elif self.algorithm == 'rfb':
            model = self.train_random_forest_bagging()

        elif self.algorithm == 'lrg':
            model = self.train_linear_regression()

        elif self.algorithm == 'all':
            model1, model2, model3 = self.train_all()
            model = model1

        return model

    def train_knn(self):

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        
        knn_model = KNeighborsRegressor(n_neighbors=20, metric='euclidean')
        knn_model.fit(X_train, y_train)
        y_pred_train = knn_model.predict(X_train)
        y_pred_test = knn_model.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print(f"Mean Squared Error (Train) [KNN]: {mse_train}")
        print(f"Mean Squared Error (Test) [KNN]: {mse_test}")

        return knn_model
    
    def train_linear_regression(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_train = lr_model.predict(X_train)
        y_pred_test = lr_model.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print(f"Mean Squared Error (Train) [Linear]: {mse_train}")
        print(f"Mean Squared Error (Test) [Linear]: {mse_test}")

        return lr_model
    
    def train_random_forest_bagging(self):

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        rf_model = RandomForestRegressor(n_estimators=100, max_features='sqrt', n_jobs=-1)  # Adjust max_features as needed
        rf_model.fit(X_train, y_train)
        mse_train_bagging = mean_squared_error(y_train, rf_model.predict(X_train))
        mse_test_bagging = mean_squared_error(y_test, rf_model.predict(X_test))
        print(f"Mean Squared Error (Train) [Bagging (Random Forest)]: {mse_train_bagging}")
        print(f"Mean Squared Error (Test) [Bagging (Random Forest)]: {mse_test_bagging}")
        return rf_model
    
    def train_all(self):

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        knn_model = self.train_knn()
        lrg_model = self.train_linear_regression()
        rfb_model  = self.train_random_forest_bagging()
        
        return knn_model, lrg_model, rfb_model
