import argparse
from extractor import FeatureExtractor
from file_parser import get_audio_file_paths, print_file_paths
from model_trainer import ModelTrainer, RNNModel
from reconstructor import reconstruct_audio, display_spectrograms
from logger.logger import log
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import soundfile as sf
from scipy.signal import wiener
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotter

parser = argparse.ArgumentParser(description='A simple command-line argument parser example.')
parser.add_argument('--data_path','-data' , type=str, help='path to dataset', default='./sample-data/')
parser.add_argument('--max_audio_length', '-max', type=int, help='Max audio length of your entire dataset', default=4)
parser.add_argument('--sample_rate', '-sr', type=int, help='Sample rate of audio', default=22050)
parser.add_argument('--algo', '-a', choices=['knn', 'rfb', 'lrg', 'cnn', 'rnn', 'k_r_l', 'cnn_rnn'], help='Learning algorithm', default='knn')
parser.add_argument('--eval_or_test', '-e', choices=['eval', 'test'], help='Whether you are testnig or evaluating', default='test')
parser.add_argument('--input','-i' , type=str, help='path to input', default='./sample-data/sm3_n0L.wav')
parser.add_argument('--gender', '-g', choices=['male', 'female', 'both'], help='Gender of person speaking', default='male')
parser.add_argument('--sound_type', '-st', choices=['sentence', 'digit', 'both'], help='Speaking digits or sentences', default='sentence')
parser.add_argument('--neighbors', '-n', type=int, help='Number of neighbors for KNN', default=5)
parser.add_argument('--estimators', '-est', type=int, help='Number of estimators for RFB', default=100)
parser.add_argument('--hidden_size', '-hs', type=int, help='Hidden size for rnn', default=32)
parser.add_argument('--batch_size', '-bs', type=int, help='Batch Size', default=4)
parser.add_argument('--epochs', '-epo', type=int, help='Number of Epochs', default=10)
parser.add_argument('--num_rnn_layers', '-nl', type=int, help='Number of layers for rnn', default=4)


args = parser.parse_args()


data_path = args.data_path
max_audio_length = args.max_audio_length
algo = args.algo
input_file = args.input
exclude_file = input_file
eval_or_test =  args.eval_or_test
model = None
model2 = None
model3 = None
sound_type = args.sound_type
gender = args.gender
sample_rate = args.sample_rate
neighbors = args.neighbors
estimators = args.estimators
epochs = args.epochs
num_rnn_layers = args.num_rnn_layers
hidden_size = args.hidden_size
batch_size = args.batch_size


if algo == 'knn':
    with open('./models/knn_model.pkl', 'rb') as f:
        model = pickle.load(f)
elif algo == 'rfb':
    with open('./models/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
elif algo == 'lrg':
    with open('./models/lrg_model.pkl', 'rb') as f:
        model = pickle.load(f)
elif algo =='k_r_l':
    with open('./models/knn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./models/rf_model.pkl', 'rb') as f:
        model2 = pickle.load(f)
    with open('./models/lrg_model.pkl', 'rb') as f:
        model3 = pickle.load(f)
    
elif algo == 'cnn':
    pass
elif algo == 'rnn':
    pass


 # Get file paths of sample data set
log('Cmd params: ' + str(args))
file_paths = get_audio_file_paths(data_path, gender, sound_type, exclude_file)
print_file_paths(file_paths)

# Extract features and Labels
feature_extractor = FeatureExtractor(file_paths, sample_rate, max_audio_length)
x, y = feature_extractor.extract_features()
log('Finished feature extraction')
log('X (noisy) features shape: ' + str(x.shape))
log('Y (clean) labels shape: ' + str(y.shape))

# Extract Features from selected audio file
x_input, input_shape = feature_extractor.extract_features_single(input_file)
log('Input x (noisy) feature shape:' + str(x_input.shape))
y_target = None

filename = str(algo) + '_target_audio.wav'

# If you are just testing
if eval_or_test == 'test':

    if algo == 'cnn':

        model_trainer = ModelTrainer(x, y, algo, neighbors, estimators, epochs, num_rnn_layers, hidden_size, batch_size)
        model = load_model('./models/cnn_model.h5')

        x_input = x_input.reshape(1, -1, 1)
        y_target = model.predict(x_input).flatten().reshape(input_shape)

        log("Finished prediction. Y_label: " + str(y_target.shape))

    elif algo == 'rnn':
      
        # Load previosly trained model
        model_trainer = ModelTrainer(x, y, algo, neighbors, estimators, epochs, num_rnn_layers, hidden_size, batch_size)
        model_trainer.x = model_trainer.x.reshape((model_trainer.x.shape[0], model_trainer.x.shape[1], 1))
        
        x_train_tensor = torch.from_numpy(model_trainer.x).float()
        y_train_tensor = torch.from_numpy(model_trainer.y).float()
        
        model = RNNModel(input_size=x_train_tensor.shape[2],  hidden_size =hidden_size, num_layers =num_rnn_layers,  output_size=y_train_tensor.shape[1])
        model.load_state_dict(torch.load('./models/rnn_model.ckpt'))
        
        # Set in eval mode and move to gpu if available
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Convert to PyTorch tensor
        x_input = torch.tensor(x_input, dtype=torch.float32).unsqueeze(-1)  # Ensure it is [sequence_length, 1]
        x_input = x_input.unsqueeze(0).to(device)  # Add batch dimension [1, sequence_length, 1]

        # Get y-label prediction
        with torch.no_grad():
            y_pred = model(x_input) 

        # Post proccessing
        y_target = y_pred.cpu().numpy().flatten().reshape(input_shape)
        x_input = x_input.cpu().numpy().reshape(input_shape)

    # if knn, lrg, or rfb
    else:
        log("Testing model using input file: " + str(input_file))
        y_target = model.predict([x_input]).flatten().reshape(input_shape)
        log("Predicted y_target of shape " + str(y_target.shape) + " using "  + str(model))

    # Save Spectrogram
    log('Starting to create sepctrogram')
    display_spectrograms(x_input.reshape(input_shape), y_target, sample_rate)
    log('Finished saving sepctrogram')

    # Reconstuct Audio
    log('Starting to reconstruct audio')
    reconstruct_audio(y_target, sample_rate, filename)
    log('Finished reconstructing audio')

    # Apply Weiner filter
    data, samplerate = sf.read(filename)
    filtered_data = wiener(data)
    sf.write(filename, filtered_data, samplerate)
    log('Applied Weiner filter')

# If you are evaluating
else:
    if algo == 'cnn':
        model_trainer = ModelTrainer(x, y, algo, neighbors, estimators, epochs, num_rnn_layers, hidden_size, batch_size)
        model = load_model('./models/cnn_model.h5')

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
       
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('CNN Training and Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('./evaluations/cnn_loss.png')
        plt.show()

    elif algo == 'rnn':
        # Load previosly trained model
        model_trainer = ModelTrainer(x, y, algo, neighbors, estimators, epochs, num_rnn_layers, hidden_size, batch_size)
        model_trainer.x = model_trainer.x.reshape((model_trainer.x.shape[0], model_trainer.x.shape[1], 1))
        
        x_train_tensor = torch.from_numpy(model_trainer.x).float()
        y_train_tensor = torch.from_numpy(model_trainer.y).float()

        # Create the DataLoader for batch processing
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=model_trainer.batch_size, shuffle=True)

        model = RNNModel(input_size=x_train_tensor.shape[2],  hidden_size =hidden_size, num_layers =num_rnn_layers,  output_size=y_train_tensor.shape[1])
        model.load_state_dict(torch.load('./models/rnn_model.ckpt'))
        
        # Set in eval mode and move to gpu if available
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        
        model.train()
        # Adjust the training loop with gradient clipping
        for epoch in range(model_trainer.epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
                optimizer.step()
                
                model_trainer.losses.append(loss.item())
                print(f'Epoch [{epoch+1}/{model_trainer.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
            
        # Save the model checkpoint
        torch.save(model.state_dict(), './models/rnn_model.ckpt')
        # Plotting the performance over different training sizes

        
        iterations_per_epoch = len(model_trainer.losses) // epochs
        rnn_epoch_losses = [model_trainer.losses[i] for i in range(0, len(model_trainer.losses), iterations_per_epoch)] # Sample decreasing loss

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), rnn_epoch_losses, label='Train Loss')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.title('RNN Regression Model Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig('./evaluations/rnn_loss.png')
        plt.show()
        
    elif algo =='cnn_rnn':
        model_trainer = ModelTrainer(x, y, algo, neighbors, estimators, epochs, num_rnn_layers, hidden_size, batch_size)
        model = load_model('./models/cnn_model.h5')

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

        cnn_loss =  history.history['loss']


        # Load previosly trained model
        model_trainer = ModelTrainer(x, y, algo, neighbors, estimators, epochs, num_rnn_layers, hidden_size, batch_size)
        model_trainer.x = model_trainer.x.reshape((model_trainer.x.shape[0], model_trainer.x.shape[1], 1))
        
        x_train_tensor = torch.from_numpy(model_trainer.x).float()
        y_train_tensor = torch.from_numpy(model_trainer.y).float()

        # Create the DataLoader for batch processing
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=model_trainer.batch_size, shuffle=True)

        model = RNNModel(input_size=x_train_tensor.shape[2],  hidden_size =hidden_size, num_layers =num_rnn_layers,  output_size=y_train_tensor.shape[1])
        model.load_state_dict(torch.load('./models/rnn_model.ckpt'))
        
        # Set in eval mode and move to gpu if available
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        total_iterations = 0
        model.train()
        # Adjust the training loop with gradient clipping
        for epoch in range(model_trainer.epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
                optimizer.step()
                
                model_trainer.losses.append(loss.item())
                print(f'Epoch [{epoch+1}/{model_trainer.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
                total_iterations += 1

        iterations_per_epoch = len(model_trainer.losses) // epochs
        rnn_epoch_losses = [model_trainer.losses[i] for i in range(0, len(model_trainer.losses), iterations_per_epoch)] # Sample decreasing loss


        plotter.plot_cnn_rnn(epochs, cnn_loss, rnn_epoch_losses)
        

    elif algo == 'k_r_l':
        plotter.plot_knn_rfb_linear(x,y,model, model2,model3)

    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        plotter.plot_mse(y_pred_train, y_pred_test, y_train, y_test, algo)
        plotter.plot_rmse(y_pred_train, y_pred_test, y_train, y_test, algo)
        plotter.plot_r2(y_pred_train, y_pred_test, y_train, y_test, algo)

   

       

       
       
