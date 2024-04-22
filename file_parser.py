"""Responisble for getting the file paths in the location of the data set"""
from logger.logger import log

def get_audio_file_paths(dataset_path, gender, sound_type, excluded_file):
    """
    Args:
        dataset_path (str): path to data
        gender (str): gender of person in audio
        sound_type (str): digit or sentence in audio file
        excluded_file (str): file to exclude from dataset

    Returns: [(str,str)]: list of noisy and clean audio paths
    """
    print("Excluding file:", excluded_file)

    list_of_samples = []
    genders = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3']
    if gender == 'male':
        genders = ['m1', 'm2', 'm3']
    elif gender == 'female':
        genders = ['f1', 'f2', 'f3']

    # Helper function to process each gender
    def process_files(base_utterance, type_prefix):
        for filter_version in ['fi1', 'fi2', 'fi3', 'fi4']:
            for noise_level in ['n0L', 'n0H', 'n1L', 'n1H', 'n2L', 'n2H', 'r1H', 'r1L', 'r2H', 'r2L']:
                filtered_file = f'{dataset_path}{type_prefix}{base_utterance}_{filter_version}.wav'
                noisy_file = f'{dataset_path}{type_prefix}{base_utterance}_{noise_level}.wav'
                if noisy_file != excluded_file:
                    list_of_samples.append((noisy_file, filtered_file))

    # Determine sound type and process accordingly
    if sound_type == 'sentence':
        for base_utterance in genders:
            process_files(base_utterance, 's')
    elif sound_type == 'digit':
        for base_utterance in genders:
            process_files(base_utterance, 'd')
    else:  # Assume both digits and sentences if not specified
        for base_utterance in genders:
            process_files(base_utterance, 's')
            process_files(base_utterance, 'd')

    return list_of_samples

def print_file_paths(file_paths):
    """
    Prints paths to audio files
    Args:file_paths ([(str, str)]): file paths
    """
    print("====================================================")
    for i,j in file_paths:
        print(i,j)
    print("====================================================")
    print("Number of audio files: ", len(file_paths))
    print("====================================================")