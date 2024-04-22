"""Responsible for extracting noisy and clean features and lables from audio"""

import librosa
import numpy as np
from logger.logger import log
import numpy as np

class FeatureExtractor:

    """Extracts features from an audio file"""

    def __init__(self, f_paths, default_sr, max_audio_length):
        """
        init FeatureExtractor

        Args:
            f_paths ([(str, str)]]): file paths for noisy and clean audio
            default_sr (int): sample rate
            max_audio_length (int): max audio length of file in seconds
        """

        self.t_extracted = 0
        self.f_paths = f_paths
        self.sr = default_sr
        self.mal = max_audio_length
        self.samples = [] 
        self.features = []
        self.labels = []
        self.FMIN = 20

    def extract_features_single(self, file_path):
        """Extracts melspectrogram from a given audio file
        
        Args: 
            file_path (str): file path
        
        Returns:
            ndarray, _Shape: feature vector, melspect shape
        """
        # Load noisy audio file
        y_noisy, sr_noisy = librosa.load(file_path)

        # Pad audio file length
        y_noisy_padded = librosa.util.fix_length(data=y_noisy, size=self.sr*self.mal)

        # Extract melspectrogram feature and keep track of shape
        melspectrogram_feature = librosa.feature.melspectrogram(y=y_noisy_padded, sr=self.sr)
        melspectrogram_shape = melspectrogram_feature.shape
        
        # Flatten matrix and return feature vector and shape of melspectrogram feature
        feature_vector = melspectrogram_feature.flatten()
        return feature_vector, melspectrogram_shape 

    def extract_features(self):
        """Extracts features from noisy and clean audio files and puts them into 
        a vector of feature vectors and a vector of label vectors.

        Returns:
            ndarray, ndarray: feature vector, label vector
        """

        for n_file, f_file in self.f_paths:
            
            # Extract noisy audio, pad file, extract mel feature
            y_noisy, sr_noisy = librosa.load(n_file, sr=self.sr)
            y_noisy_padded = librosa.util.fix_length(data=y_noisy, size=self.sr*self.mal)
            melspectrogram_noisy = librosa.feature.melspectrogram(y=y_noisy_padded, sr=self.sr)

            # Extract filtered audio, pad file, extract mel feature
            y_filtered, sr_filtered = librosa.load(f_file, sr=self.sr)
            y_filtered_padded = librosa.util.fix_length(data=y_filtered, size=self.sr*self.mal)             
            melspectrogram_filtered = librosa.feature.melspectrogram(y=y_filtered_padded, sr=self.sr)

            # Noisy files comprise feature vector
            self.features.append(melspectrogram_noisy.flatten())

            # Clean files comprise label vector
            self.labels.append(melspectrogram_filtered.flatten())

            self.t_extracted += 1
            print(f"Finished extracting: {n_file} {f_file} | {round(self.t_extracted/len(self.f_paths), 4)}% COMPLETE")

        return np.array(self.features), np.array(self.labels)

# EXAMPLE USSAGE
if __name__ == "__main__":
    feature_extractor = FeatureExtractor('./sample-data/', 22050, 4)
    print(feature_extractor)
