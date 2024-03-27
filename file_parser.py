"""Responisble for getting the file paths in the location of the data set"""

def get_audio_file_paths(dataset_path, gender, sound_type):

    list_of_samples = []
    genders = ['sf1', 'sf2', 'sf3', 'sm1', 'sm2', 'sm3']

    if gender == 'male':
        genders = ['sm1', 'sm2', 'sm3']
    elif gender == 'female':
        genders = ['sf1', 'sf2', 'sf3']

    if sound_type == 'sentence':
        for base_utterance_sent in genders:
            for filter_version in ['fi1']:
                for noise_level in ['n0L', 'n0H', 'n1L', 'n1H', 'n2L', 'n2H', 'r1L', 'r1H', 'r2L', 'r2H']:
                    filtered_file_sent = f'{dataset_path}{base_utterance_sent}_{filter_version}.wav'
                    noisy_file_sent = f'{dataset_path}{base_utterance_sent}_{noise_level}.wav'
                    list_of_samples.append((noisy_file_sent, filtered_file_sent))
    elif sound_type == 'digit':
        for base_utterance_digit in genders:
            for filter_version in ['fi1']:
                for noise_level in ['n0L', 'n0H', 'n1L', 'n1H', 'n2L', 'n2H', 'r1L', 'r1H', 'r2L', 'r2H']:
                    filtered_file_digit = f'{dataset_path}{base_utterance_digit}_{filter_version}.wav'
                    noisy_file_digit = f'{dataset_path}{base_utterance_digit}_{noise_level}.wav'
                    list_of_samples.append((noisy_file_digit, filtered_file_digit))
    else:
        # If neither 'sentence' nor 'digit', assume both and include both types
        for base_utterance_sent in genders:
            for filter_version in ['fi1']:
                for noise_level in ['n0L', 'n0H', 'n1L', 'n1H', 'n2L', 'n2H', 'r1L', 'r1H', 'r2L', 'r2H']:
                    filtered_file_sent = f'{dataset_path}{base_utterance_sent}_{filter_version}.wav'
                    noisy_file_sent = f'{dataset_path}{base_utterance_sent}_{noise_level}.wav'
                    list_of_samples.append((noisy_file_sent, filtered_file_sent))

        for base_utterance_digit in genders:
            for filter_version in ['fi1']:
                for noise_level in ['n0L', 'n0H', 'n1L', 'n1H', 'n2L', 'n2H', 'r1L', 'r1H', 'r2L', 'r2H']:
                    filtered_file_digit = f'{dataset_path}{base_utterance_digit}_{filter_version}.wav'
                    noisy_file_digit = f'{dataset_path}{base_utterance_digit}_{noise_level}.wav'
                    list_of_samples.append((noisy_file_digit, filtered_file_digit))

    return list_of_samples


def print_file_paths(file_paths):
    print("====================================================")
    for i,j in file_paths:
        print(i,j)
    print("====================================================")
    print("Number of audio files: ", len(file_paths))
    print("====================================================")