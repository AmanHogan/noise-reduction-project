import numpy as np
from logger.logger import log
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input, Dense, Dropout, BatchNormalization,Input,Dense,Input
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import plotter

class RNNModel(nn.Module):
    """Class for creating RNN"""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.3) 

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out) 
        return out

class ModelTrainer():
    """Trains a model using s specified alforithm on the audio dataset"""

    def __init__(self, x, y, algorithm, neighbors, estimators, epochs, num_rnn_layers, hidden_size, batch_size):
        """init

        Args:
            x (ndarray): features
            y (ndarray): labels
            algorithm (str): algorithm
            neighbors (int): number of neighbores for knn
            estimators (int): num of estimators in rfb
            epochs (int): epochs for knn and cnn
            num_rnn_layers (int): num of layers in rnn
            hidden_size (int): size of layer
            batch_size (int): batch size for cnn and rnn
        """

        self.x = x
        self.y = y
        self.algorithm = algorithm
        self.knn_n = neighbors
        self.estimators = estimators
        self.losses = []
        self.epochs = epochs
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def run_trainer(self):
        """Runs specified algorithm

        Returns:
            Model : model chosen
        """
        model = None
        if self.algorithm == 'knn':
            model = self.train_knn()
        elif self.algorithm == 'rfb':
            model = self.train_rfb()
        elif self.algorithm == 'lrg':
            model = self.train_lrg()
        elif self.algorithm == 'cnn':
            model = self.train_cnn()
        elif self.algorithm == 'rnn':
            model = self.train_rnn()
        return model

    def train_knn(self):
        """Trains model using KNN regression

        Returns:
            model: knn regression model
        """

        # Load data and scale the data
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        max_knn_iterations = 15
        
        # Create model
        knn_model = KNeighborsRegressor(n_neighbors=self.knn_n, metric='euclidean')

        # Tracking performance for elbow plot
        train_errors, test_errors = [], []
        
        # Create errors for an elbow plot
        log("Creating Elbow Plot for knn")
        for i in range(1, max_knn_iterations): 
            
            # Fit model
            knn_model.n_neighbors = i
            knn_model.fit(X_train_scaled, y_train)

            # Evaluation metrics
            y_train_pred = knn_model.predict(X_train_scaled)
            y_test_pred = knn_model.predict(X_test_scaled)
            train_errors.append(mean_squared_error(y_train, y_train_pred))
            test_errors.append(mean_squared_error(y_test, y_test_pred))
            log("Training KNN " + str(i) + " with a loss of " + str(train_errors[i-1]))

        # Plot elbow
        plotter.plot_elbow(train_errors, test_errors)

        # Create Knn model with specified neibghores
        knn_model.n_neighbors = self.knn_n
        knn_model.fit(X_train_scaled, y_train)
    
        # Save Model
        with open('./models/knn_model.pkl','wb') as f:
            pickle.dump(knn_model,f)

        return knn_model
    
    def train_lrg(self):
        """Trains model using Linear Regression
        
        Returns:
            model: model using Linear Regression
        """

        # Load data
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create Model
        lrg_model = LinearRegression()
        lrg_model.fit(X_train_scaled, y_train)

        # Evaluation metrics
        final_mse_train = mean_squared_error(y_train, lrg_model.predict(X_train_scaled))
        final_mse_test = mean_squared_error(y_test, lrg_model.predict(X_test_scaled))
        print(f"Final Mean Squared Error (Train): {final_mse_train}")
        print(f"Final Mean Squared Error (Test): {final_mse_test}")

        # Save Model
        with open('./models/lrg_model.pkl','wb') as f:
            pickle.dump(lrg_model, f)

        return lrg_model

    def train_cnn(self):
        """Trains CNN using training data

        Returns:
            Model: cnn Model
        """

        # Preprocess input data
        self.x = self.x.reshape((self.x.shape[0], self.x.shape[1], 1))
        input_shape = (self.x.shape[1], 1)
        y_train = self.y 

        # 1 x 32 x 3, Pool 2, Relu
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv1D(32, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        # 1 x 64 x 3, Pool 2, Relu
        model.add(Conv1D(64, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        # 1 x 128 x 3, Relu
        model.add(Conv1D(128, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling1D())

        # Dense Layer
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.y.shape[1], activation='linear'))
        
        # Fit model
        X_train = self.x.reshape((self.x.shape[0], self.x.shape[1], 1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)

        # Save model
        model.save("./models/cnn_model.h5")

        return model

    def train_rnn(self):
        """Train an RNN on the trainin data

        Returns:
            RNN Model: rnn model
        """
        
        # Preprocess the input data into tensors
        self.x = self.x.reshape((self.x.shape[0], self.x.shape[1], 1))
        x_train_tensor = torch.from_numpy(self.x).float()
        y_train_tensor = torch.from_numpy(self.y).float()

        # Create data loaders
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        # Initialize the model and move it to the GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RNNModel(input_size=x_train_tensor.shape[2], hidden_size=self.hidden_size, num_layers=self.num_rnn_layers, output_size=y_train_tensor.shape[1])
        model.to(device)

        # Initialize loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train Model
        model.train()
        for epoch in range(self.epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                
                # Prediction
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Loss and backward pass
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                
                self.losses.append(loss.item())
                print(f'Epoch [{epoch+1}/{self.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
            
        # Save the model checkpoint
        torch.save(model.state_dict(), './models/rnn_model.ckpt')
        plotter.plot_rnn(self.losses)

        
        return model
    
    def train_rfb(self):
        """Trains model using Random Forest Bagging
        
        Returns: 
            model: model using Random Forest Bagging
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

        # Save model
        with open('./models/rf_model.pkl','wb') as f:
            pickle.dump(rf_model,f)

        return rf_model
    
