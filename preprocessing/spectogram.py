import os
from re import I
import sys
from tkinter import W
import numpy as np
import librosa

# import librsoa
sys.path.append("..")


def convert_to_mel_spectrum(data: np.ndarray, epsilon:float=1e-8):
    """Converts data to mel spectrum
    Args:
        data (np.ndarray) : data to convert
    Returns:
        spectrum (np.ndarray) : 2d array of spectrum
    """
    return np.log10(librosa.feature.melspectrogram(y=data)+epsilon)


def convert_to_mfcc(data: np.ndarray):
    """Converts data to mfcc
    Args:
        data (np.ndarray) : data to convert
    Returns:
        mfcc (np.ndarray) : 2d array of mfcc
    """
    return librosa.feature.mfcc(y=data)


def convert_mel_spectrum_to_wav(data: np.ndarray):
    """Converst mel spectrum to wav
    Args:
        data (np.ndarray) : data to convert
    Returns:
        wav (np.ndarray) : 1d array of wav
    """
    return librosa.feature.inverse.mel_to_audio(data)

def convert_mfcc_to_wav(data: np.ndarray):
    """Converst mfcc to wav
    Args:
        data (np.ndarray) : data to convert
    Returns:
        wav (np.ndarray) : 1d array of wav
    """
    return librosa.feature.inverse.mfcc_to_audio(data)


if __name__ == "__main__":
    test_data = np.random.rand(22050)
    print("test data shape", test_data.shape)

    mel_spectrum = convert_to_mel_spectrum(test_data)
    print("mel spectrum shape", mel_spectrum.shape)
