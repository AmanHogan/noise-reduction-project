"""Reconstructs audio and displays spectrograms"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from logger.logger import log
import soundfile as sf

def reconstruct_audio(y, sr, filename):
    """Reconstruct audio using new clean spectrogram

    Args:
        y (ndarray): target melspectrogram
        sr (int): sample rate
    """

    # Convert Mel spectrogram to linear spectrogram (STFT magnitude)
    linear_spectrogram = librosa.feature.inverse.mel_to_stft(y)
    y_reconstructed = librosa.griffinlim(linear_spectrogram)
    sf.write(filename, y_reconstructed, sr)

def display_spectrograms(x, y, sr):
    """Display noisy and clean audio files

    Args:
        x (ndarray): noisy audio file input
        y (ndarray): target audio file output
        sr (int): sample rate
    """
    
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(x, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Noisy Audio')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(y, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Reconstructed Audio')
    plt.tight_layout()

    plt.savefig('spectrograms.png')


 


