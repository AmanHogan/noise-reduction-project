"""Responisble for getting the file paths in the location of the data set"""
from logger.logger import log

def get_audio_file_paths(dataset_path, gender, sound_type):
    """
    f<#> -  A filtered audio under a given channel\n
    <m/f> - Male or female audio\n
    <d/s> - digits or sentences\n
    <0/1/2> <H/L> - Type of noise applied and how much noise\n
    <r/n> - reverb or noise\n

    Args:
        dataset_path (str): path to data
        gender (str): gender of person in audio
        sound_type (str): digit or senence in audio file
        Gets the list of noisy and filtered audio files.

    Returns: [(str,str)]: list of noisy and clean audio paths
    """

    list_of_samples = []

    # Specifiy which gender you want for the audio file
    genders = ['f1', 'f2', 'f3','m1', 'm2', 'm3']
    if gender == 'male':
        genders = ['m1', 'm2', 'm3']
    elif gender == 'female':
        genders = ['f1', 'f2', 'f3']

    # Get audio file for sentences
    if sound_type == 'sentence':
        for base_utterance_sent in genders:
            for filter_version in ['fi1', 'fi2', 'fi3', 'fi4']:
                for noise_level in ['n0L', 'n0H', 'n1L', 'n1H', 'n2L', 'n2H', 'r1H', 'r1L', 'r2H', 'r2L']:
                    filtered_file_sent = f'{dataset_path}s{base_utterance_sent}_{filter_version}.wav'
                    noisy_file_sent = f'{dataset_path}s{base_utterance_sent}_{noise_level}.wav'
                    list_of_samples.append((noisy_file_sent, filtered_file_sent))

    # Get audio file for digits
    elif sound_type == 'digit':
        for base_utterance_digit in genders:
            for filter_version in ['fi1', 'fi2', 'fi3', 'fi4']:
                for noise_level in ['n0L', 'n0H', 'n1L', 'n1H', 'n2L', 'n2H',  'r1H', 'r1L', 'r2H', 'r2L']:
                    filtered_file_digit = f'{dataset_path}d{base_utterance_digit}_{filter_version}.wav'
                    noisy_file_digit = f'{dataset_path}d{base_utterance_digit}_{noise_level}.wav'
                    list_of_samples.append((noisy_file_digit, filtered_file_digit))
    
    # Get audio files for digits and sentences
    else:
        for base_utterance_sent in genders:
            for filter_version in ['fi1', 'fi2', 'fi3', 'fi4']:
                for noise_level in ['n0L', 'n0H', 'n1L', 'n1H', 'n2L', 'n2H',  'r1H', 'r1L', 'r2H', 'r2L']:
                    filtered_file_sent = f'{dataset_path}s{base_utterance_sent}_{filter_version}.wav'
                    noisy_file_sent = f'{dataset_path}s{base_utterance_sent}_{noise_level}.wav'
                    list_of_samples.append((noisy_file_sent, filtered_file_sent))

        for base_utterance_digit in genders:
            for filter_version in ['fi1', 'fi2', 'fi3', 'fi4']:
                for noise_level in ['n0L', 'n0H', 'n1L', 'n1H', 'n2L', 'n2H',  'r1H', 'r1L', 'r2H', 'r2L']:
                    filtered_file_digit = f'{dataset_path}d{base_utterance_digit}_{filter_version}.wav'
                    noisy_file_digit = f'{dataset_path}d{base_utterance_digit}_{noise_level}.wav'
                    list_of_samples.append((noisy_file_digit, filtered_file_digit))

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