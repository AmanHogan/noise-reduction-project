import os
import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, f_paths, default_sr, max_audio_length):
        self.total_extracted = 0
        self.f_paths = f_paths
        self.default_sr = default_sr
        self.max_audio_length = max_audio_length
        self.samples = []
        self.features = []
        self.labels = []
        self.f_min = 20
        
    
    def extract_features(self, feat_types):
        if feat_types == 'all':
            for noisy_file, filtered_file in self.f_paths:

                y_noisy, sr_noisy = librosa.load(noisy_file, sr=self.default_sr)
                y_noisy = librosa.util.fix_length(data=y_noisy, size=self.default_sr*self.max_audio_length)  # Fix length to 2048 samples

                y_filtered, sr_filtered = librosa.load(filtered_file, sr=self.default_sr)
                y_filtered = librosa.util.fix_length(data=y_filtered, size=self.default_sr*self.max_audio_length) 
                
                # Spectral Features
                chroma_stft_noisy = librosa.feature.chroma_stft(y=y_noisy, sr=sr_noisy)
                chroma_cens_noisy = librosa.feature.chroma_cens(y=y_noisy, sr=sr_noisy, fmin=self.f_min)
                melspectrogram_noisy = librosa.feature.melspectrogram(y=y_noisy, sr=sr_noisy)
                mfcc_noisy = librosa.feature.mfcc(y=y_noisy, sr=sr_noisy)
                rmse_noisy = librosa.feature.rms(y=y_noisy)

                chroma_stft_filtered = librosa.feature.chroma_stft(y=y_filtered, sr=sr_filtered)
                chroma_cens_filtered = librosa.feature.chroma_cens(y=y_filtered, sr=sr_filtered, fmin=self.f_min)
                melspectrogram_filtered = librosa.feature.melspectrogram(y=y_filtered, sr=sr_filtered)
                mfcc_filtered = librosa.feature.mfcc(y=y_filtered, sr=sr_filtered)
                rmse_filtered = librosa.feature.rms(y=y_filtered)

                # Spectral Centroid Features
                spectral_centroid_noisy = librosa.feature.spectral_centroid(y=y_noisy, sr=sr_noisy)
                spectral_centroid_filtered = librosa.feature.spectral_centroid(y=y_filtered, sr=sr_filtered)

                # Spectral Bandwidth Features
                spectral_bandwidth_noisy = librosa.feature.spectral_bandwidth(y=y_noisy, sr=sr_noisy)
                spectral_bandwidth_filtered = librosa.feature.spectral_bandwidth(y=y_filtered, sr=sr_filtered)

                # Spectral Contrast Features
                spectral_contrast_noisy = librosa.feature.spectral_contrast(y=y_noisy, sr=sr_noisy, fmin=self.f_min)
                spectral_contrast_filtered = librosa.feature.spectral_contrast(y=y_filtered, sr=sr_filtered, fmin=self.f_min)

                # Spectral Flatness Features
                spectral_flatness_noisy = librosa.feature.spectral_flatness(y=y_noisy)
                spectral_flatness_filtered = librosa.feature.spectral_flatness(y=y_filtered)

                # Spectral Rolloff Features
                spectral_rolloff_noisy = librosa.feature.spectral_rolloff(y=y_noisy, sr=sr_noisy)
                spectral_rolloff_filtered = librosa.feature.spectral_rolloff(y=y_filtered, sr=sr_filtered)

                # Poly Features
                poly_features_noisy = librosa.feature.poly_features(y=y_noisy, sr=sr_noisy)
                poly_features_filtered = librosa.feature.poly_features(y=y_filtered, sr=sr_filtered)

                # Tonnetz Features
                tonnetz_noisy = librosa.feature.tonnetz(y=y_noisy, sr=sr_noisy, fmin=self.f_min)
                tonnetz_filtered = librosa.feature.tonnetz(y=y_filtered, sr=sr_filtered, fmin=self.f_min)

                # Zero Crossing Rate Features
                zcr_noisy = librosa.feature.zero_crossing_rate(y=y_noisy)
                zcr_filtered = librosa.feature.zero_crossing_rate(y=y_filtered)
                
                # Concatenate feature matrices for noisy and filtered audio
                feature_vector_noisy = np.concatenate((chroma_stft_noisy.flatten(), chroma_cens_noisy.flatten(), melspectrogram_noisy.flatten(), 
                                                    mfcc_noisy.flatten(), rmse_noisy.flatten(), spectral_centroid_noisy.flatten(),
                                                    spectral_bandwidth_noisy.flatten(), spectral_contrast_noisy.flatten(),
                                                    spectral_flatness_noisy.flatten(), spectral_rolloff_noisy.flatten(),
                                                    poly_features_noisy.flatten(), tonnetz_noisy.flatten(), zcr_noisy.flatten()))
                
                feature_vector_filtered = np.concatenate((chroma_stft_filtered.flatten(), chroma_cens_filtered.flatten(), melspectrogram_filtered.flatten(), 
                                                        mfcc_filtered.flatten(), rmse_filtered.flatten(), spectral_centroid_filtered.flatten(),
                                                        spectral_bandwidth_filtered.flatten(), spectral_contrast_filtered.flatten(),
                                                        spectral_flatness_filtered.flatten(), spectral_rolloff_filtered.flatten(),
                                                        poly_features_filtered.flatten(), tonnetz_filtered.flatten(), zcr_filtered.flatten()))

                # Combine feature vectors into one sample and append to samples list
                sample = np.concatenate((feature_vector_noisy, feature_vector_filtered))
                self.samples.append(sample)
                self.features.append(feature_vector_filtered)
                self.labels.append(feature_vector_noisy)
                self.total_extracted = self.total_extracted + 1
                print(f"Finished sample: {noisy_file} {filtered_file} | PERCENT DONE: {round(self.total_extracted/len(self.f_paths), 4)} %")

            return np.array(self.features), np.array(self.labels)

        if feat_types == 'some':
            for noisy_file, filtered_file in self.f_paths:

                y_noisy, sr_noisy = librosa.load(noisy_file, sr=self.default_sr)
                y_noisy = librosa.util.fix_length(data=y_noisy, size=self.default_sr*self.max_audio_length)  # Fix length to 2048 samples

                y_filtered, sr_filtered = librosa.load(filtered_file, sr=self.default_sr)
                y_filtered = librosa.util.fix_length(data=y_filtered, size=self.default_sr*self.max_audio_length) 
                
                # Spectral Features
                mfcc_noisy = librosa.feature.mfcc(y=y_noisy, sr=sr_noisy)
                rmse_noisy = librosa.feature.rms(y=y_noisy)

                mfcc_filtered = librosa.feature.mfcc(y=y_filtered, sr=sr_filtered)
                rmse_filtered = librosa.feature.rms(y=y_filtered)

                # Spectral Centroid Features
                spectral_centroid_noisy = librosa.feature.spectral_centroid(y=y_noisy, sr=sr_noisy)
                spectral_centroid_filtered = librosa.feature.spectral_centroid(y=y_filtered, sr=sr_filtered)

                # Spectral Contrast Features
                spectral_contrast_noisy = librosa.feature.spectral_contrast(y=y_noisy, sr=sr_noisy, fmin=self.f_min)
                spectral_contrast_filtered = librosa.feature.spectral_contrast(y=y_filtered, sr=sr_filtered, fmin=self.f_min)

                # Spectral Flatness Features
                spectral_flatness_noisy = librosa.feature.spectral_flatness(y=y_noisy)
                spectral_flatness_filtered = librosa.feature.spectral_flatness(y=y_filtered)

                # Spectral Rolloff Features
                spectral_rolloff_noisy = librosa.feature.spectral_rolloff(y=y_noisy, sr=sr_noisy)
                spectral_rolloff_filtered = librosa.feature.spectral_rolloff(y=y_filtered, sr=sr_filtered)

                # Zero Crossing Rate Features
                zcr_noisy = librosa.feature.zero_crossing_rate(y=y_noisy)
                zcr_filtered = librosa.feature.zero_crossing_rate(y=y_filtered)
                
                # Concatenate feature matrices for noisy and filtered audio
                feature_vector_noisy = np.concatenate(( 
                                                    mfcc_noisy.flatten(), rmse_noisy.flatten(), spectral_centroid_noisy.flatten(),
                                                    spectral_contrast_noisy.flatten(),
                                                    spectral_flatness_noisy.flatten(), spectral_rolloff_noisy.flatten(),
                                                    zcr_noisy.flatten()))
                
                feature_vector_filtered = np.concatenate(( 
                                                        mfcc_filtered.flatten(), rmse_filtered.flatten(), spectral_centroid_filtered.flatten(),
                                                         spectral_contrast_filtered.flatten(),
                                                        spectral_flatness_filtered.flatten(), spectral_rolloff_filtered.flatten(),
                                                         zcr_filtered.flatten()))

                # Combine feature vectors into one sample and append to samples list
                sample = np.concatenate((feature_vector_noisy, feature_vector_filtered))
                self.samples.append(sample)
                self.features.append(feature_vector_filtered)
                self.labels.append(feature_vector_noisy)
                self.total_extracted = self.total_extracted + 1
                print(f"Finished sample: {noisy_file} {filtered_file} | PERCENT DONE: {round(self.total_extracted/len(self.f_paths), 4)} %")

            return np.array(self.features), np.array(self.labels)

        
# Example usage
if __name__ == "__main__":
    feature_extractor = FeatureExtractor('./sample-data/', 8000, 4)
