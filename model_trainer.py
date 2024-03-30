import os
import librosa
import numpy as np
from logger.logger import log
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


class ModelTrainer():

    """Trains a model using s specified alforithm on the audio dataset"""

    def __init__(self, x, y, algorithm, neighbors, estimators):
        """
        init ModelTrainer

        Args:
            x (ndarray): features for all samples
            y (ndarry): lables for all samples
            algorithm (str): algorithm used
            neighbors (int): number of neighbors for knn algorithm
            estimators (int): number of estimatos for rfb algorithm
        """
        self.x = x
        self.y = y
        self.algorithm = algorithm
        self.knn_n = neighbors
        self.estimators = estimators

    def run_trainer(self):
        """
        Runs a model trainer using a specified learning algorithm
        Returns: Model: learning model
        """

        model = None

        # K neareast neighbores
        if self.algorithm == 'knn':
            model = self.train_knn()

        # Random forest bagging
        elif self.algorithm == 'rfb':
            model = self.train_random_forest_bagging()

        # Linear regression
        elif self.algorithm == 'lrg':
            model = self.train_linear_regression()

        return model

    def train_knn(self):
        """
        Trains model using KNN
        Returns: model: model using KNN
        """

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)

        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train KNN with scaled features
        knn_model = KNeighborsRegressor(n_neighbors=self.knn_n, metric='euclidean')
        knn_model.fit(X_train_scaled, y_train)

        # Get evaluation of model
        y_pred_train = knn_model.predict(X_train_scaled)
        y_pred_test = knn_model.predict(X_test_scaled)

        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        print(f"Mean Squared Error (Train) [KNN]: {mse_train}")
        print(f"Mean Squared Error (Test) [KNN]: {mse_test}")
        
        return knn_model
    
    def train_linear_regression(self):
        """
        Trains model using Linear Regression
        Returns: model: model using Linear Regression
        """

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)

        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        # Get evaluation of model
        y_pred_train = lr_model.predict(X_train_scaled)
        y_pred_test = lr_model.predict(X_test_scaled)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        
        print(f"Mean Squared Error (Train) [Linear]: {mse_train}")
        print(f"Mean Squared Error (Test) [Linear]: {mse_test}")

        return lr_model
    
    def train_random_forest_bagging(self):
        """
        Trains model using Random Forest Bagging
        Returns: model: model using Random Forest Bagging
        """

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)

        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf_model = RandomForestRegressor(n_estimators=self.estimators, max_features='sqrt', n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)

        # Get evaluation of model
        mse_train_bagging = mean_squared_error(y_train, rf_model.predict(X_train_scaled))
        mse_test_bagging = mean_squared_error(y_test, rf_model.predict(X_test_scaled))

        print(f"Mean Squared Error (Train) [Bagging (Random Forest)]: {mse_train_bagging}")
        print(f"Mean Squared Error (Test) [Bagging (Random Forest)]: {mse_test_bagging}")

        return rf_model
    