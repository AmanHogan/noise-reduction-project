import argparse
from f_extractor import FeatureExtractor
from file_parser import get_audio_file_paths, print_file_paths
from model_trainer import ModelTrainer

def main():

    parser = argparse.ArgumentParser(description='A simple command-line argument parser example.')
    parser.add_argument('--data_path', type=str, help='path to dataset', default='./sample-data/')
    parser.add_argument('--max_audio_length', type=int, help='Max audio length of your entire dataset', default=4)
    parser.add_argument('--sample_rate', type=int, help='Sample rate of audio', default=8000)
    parser.add_argument('--gender', choices=['male', 'female', 'both'], help='Gender of person speaking', default='both')
    parser.add_argument('--sound_type', choices=['sentence', 'digit', 'both'], help='Speaking digits or sentences', default='both')
    parser.add_argument('--algo', choices=['knn', 'rfb', 'lrg', 'all'], help='Learning algorithm', default='lrg')
    parser.add_argument('--verbose', type=bool, help='Sample rate of audio', default=False)
    parser.add_argument('--feat_types', choices=['some', 'all'], help='The types of audio features you want to extract', default='all')
    args = parser.parse_args()
    
    data_path = args.data_path
    max_audio_length = args.max_audio_length
    sample_rate = args.sample_rate
    gender = args.gender
    sound_type = args.sound_type
    algo = args.algo
    verbose = args.verbose 
    feat_types = args.feat_types

    file_paths = get_audio_file_paths(data_path, gender, sound_type)
    print_file_paths(file_paths)

    feature_extractor = FeatureExtractor(file_paths, sample_rate, max_audio_length)
    x, y = feature_extractor.extract_features(feat_types)

    model_trainer = ModelTrainer(x, y, algo)
    model = model_trainer.run_trainer()

    # TODO: Implement audio reconstruction


if __name__ == "__main__":
    main()
