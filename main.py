import argparse
from extractor import FeatureExtractor
from file_parser import get_audio_file_paths, print_file_paths
from model_trainer import ModelTrainer, RNNModel
from reconstructor import reconstruct_audio, display_spectrograms
from logger.logger import log
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import soundfile as sf
from scipy.signal import wiener

def main():

    parser = argparse.ArgumentParser(description='A simple command-line argument parser example.')
    parser.add_argument('--data_path','-data' , type=str, help='path to dataset', default='./sample-data/')
    parser.add_argument('--max_audio_length', '-max', type=int, help='Max audio length of your entire dataset', default=4)
    parser.add_argument('--sample_rate', '-sr', type=int, help='Sample rate of audio', default=22050)
    parser.add_argument('--gender', '-g', choices=['male', 'female', 'both'], help='Gender of person speaking', default='both')
    parser.add_argument('--sound_type', '-st', choices=['sentence', 'digit', 'both'], help='Speaking digits or sentences', default='both')
    parser.add_argument('--algo', '-a', choices=['knn', 'rfb', 'lrg', 'cnn', 'rnn'], help='Learning algorithm', default='knn')
    parser.add_argument('--verbose', '-v', type=bool, help='Sample rate of audio', default=False)
    parser.add_argument('--neighbors', '-n', type=int, help='Number of neighbors for KNN', default=5)
    parser.add_argument('--estimators', '-e', type=int, help='Number of estimators for RFB', default=100)
    parser.add_argument('--input','-i' , type=str, help='path to input', default='./sample-data/df1_n0L.wav')
    parser.add_argument('--epochs', '-epo', type=int, help='Number of Epochs', default=10)
    parser.add_argument('--num_rnn_layers', '-nl', type=int, help='Number of layers for rnn', default=4)
    parser.add_argument('--hidden_size', '-hs', type=int, help='Hidden size for rnn', default=32)
    parser.add_argument('--batch_size', '-bs', type=int, help='Batch Size', default=4)
    args = parser.parse_args()
    
    data_path = args.data_path
    max_audio_length = args.max_audio_length
    sample_rate = args.sample_rate
    gender = args.gender
    sound_type = args.sound_type
    algo = args.algo
    verbose = args.verbose
    neighbors = args.neighbors
    estimators = args.estimators
    input_file = args.input
    exclude_file = input_file
    epochs = args.epochs
    num_rnn_layers = args.num_rnn_layers
    hidden_size = args.hidden_size
    batch_size = args.batch_size

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

    # Train Model
    log('Training model using: ' + str(algo))
    model_trainer = ModelTrainer(x, y, algo, neighbors, estimators, epochs, num_rnn_layers, hidden_size, batch_size)
    model = model_trainer.run_trainer()
    log('Finished training model using: ' + str(algo))

    # Extract Features from selected audio file
    x_input, input_shape = feature_extractor.extract_features_single(input_file)
    log('Input (noisy) x features shape:' + str(x_input.shape))
    y_target = None

    # Predict y-lables if CNN
    if algo == 'cnn':
        x_input = x_input.reshape(1, -1, 1)  # Adding batch dimension, and 1 as channel dimension
        y_target = model.predict([x_input]).flatten().reshape(input_shape)



    # Predict y-lables if RNN
    elif algo == 'rnn':

        # Evaluate mode, and use gpu if available
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Convert to PyTorch tensor
        x_input = torch.tensor(x_input, dtype=torch.float32).unsqueeze(-1)  
        x_input = x_input.unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            y_pred = model(x_input) 

        # Post proccessing
        y_target = y_pred.cpu().numpy().flatten().reshape(input_shape)
        x_input = x_input.cpu().numpy().reshape(input_shape)

    # if knn, lrg, rfb, predict y-labels
    else:
        y_target = model.predict([x_input]).flatten().reshape(input_shape)

    log('Finished prediction')
    log('Input file path:' + str(input_file))
    log('Target y labels shape:' + str(y_target.shape))

    # Display spectrograms for noisy and clean audio
    display_spectrograms(x_input.reshape(input_shape), y_target, sample_rate)
    log('Finished creating spectrogram')

    filename = str(algo) + '_target_audio.wav'
    # Reconstruct audio using clean melspectrogram
    reconstruct_audio(y_target, sample_rate, filename)
    log('Finished recreating audio')

    
    # Apply Weiner filter
    data, samplerate = sf.read(filename)
    filtered_data = wiener(data)
    
    sf.write(filename, filtered_data, samplerate)
    log('Finished applying weiner filter to reconstructed audio')

if __name__ == "__main__":
    log("---- Start Training ----")
    main()
    log("---- Finished Training ----")