import argparse
from extractor import FeatureExtractor
from file_parser import get_audio_file_paths, print_file_paths
from model_trainer import ModelTrainer
from reconstructor import reconstruct_audio, display_spectrograms
from logger.logger import log
import numpy as np

def main():

    parser = argparse.ArgumentParser(description='A simple command-line argument parser example.')
    parser.add_argument('--data_path','-data' , type=str, help='path to dataset', default='./sample-data/')
    parser.add_argument('--max_audio_length', '-max', type=int, help='Max audio length of your entire dataset', default=4)
    parser.add_argument('--sample_rate', '-sr', type=int, help='Sample rate of audio', default=22050)
    parser.add_argument('--gender', '-g', choices=['male', 'female', 'both'], help='Gender of person speaking', default='both')
    parser.add_argument('--sound_type', '-st', choices=['sentence', 'digit', 'both'], help='Speaking digits or sentences', default='both')
    parser.add_argument('--algo', '-a', choices=['knn', 'rfb', 'lrg'], help='Learning algorithm', default='knn')
    parser.add_argument('--verbose', '-v', type=bool, help='Sample rate of audio', default=False)
    parser.add_argument('--neighbors', '-n', type=int, help='Number of neighbors for KNN', default=1)
    parser.add_argument('--estimators', '-e', type=int, help='Number of estimators for RFB', default=100)
    parser.add_argument('--input','-i' , type=str, help='path to input', default='./sample-data/dm1_n0H.wav')
    args = parser.parse_args()

    log("START: " + '\n' + '======================================')
    
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
    log('PARAMS: ' + str(args) + '\n' + '======================================')

    # Get file paths
    file_paths = get_audio_file_paths(data_path, gender, sound_type)
    print_file_paths(file_paths)
    log('FILE PATHS:' + str(file_paths) + '\n' + '======================================')

    # Get training features and labels
    feature_extractor = FeatureExtractor(file_paths, sample_rate, max_audio_length)
    x, y = feature_extractor.extract_features()
    log('X features:' + str(x) + '\n' + '======================================')
    log('Y labels:' + str(y) + '\n' + '======================================')

    # Train model on data
    log('TRAINING MODEL USING:' + str(algo) + '\n' + '======================================')
    model_trainer = ModelTrainer(x, y, algo, neighbors, estimators)
    model = model_trainer.run_trainer()
    log('FINSIHED TRAINING MODEL using:' + str(algo) + '\n' + '======================================')

    # Run model with your noisy input file
    x_input, input_shape = feature_extractor.extract_features_single(input_file)
    y_target = model.predict([x_input]).flatten().reshape(input_shape)
    log('INPUT FILE:' + str(input_file) + '\n' + '======================================')
    log('INPUT X:' + str(x_input) + '\n' + '======================================')
    log('TARGET Y:' + str(y_target) + '\n' + '======================================')

    # Display spectrograms for noisy and clean audio
    display_spectrograms(x_input.reshape(input_shape), y_target, sample_rate)
    log('FINISHED CREATING SPECTROGRAM:' + str(y_target) + '\n' + '======================================')

    # Reconstruct audio using clean melspectrogram
    reconstruct_audio(y_target, sample_rate)
    log('FINISHED RECREATING AUDIO:' + str(y_target) + '\n' + '======================================')
    log("FINISHED: " + '\n' + '======================================')


if __name__ == "__main__":
    main()
