import os
from re import I
import sys
from tkinter import W
import numpy as np
import librosa

# import librsoa
sys.path.append("..")


def convert_to_mel_spectrum(data: np.ndarray):
    """Converts data to mel spectrum
    Args:
        data (np.ndarray) : data to convert
    Returns:
        spectrum (np.ndarray) : 2d array of spectrum
    """
    #print(data.shape)
    spectrum = librosa.feature.melspectrogram(y=data)
    spectrum = np.log10(spectrum+1e-8) # Adding 1e-8 to avoid log(0)
    return spectrum


def convert_to_mfcc(data: np.ndarray):
    """Converts data to mfcc
    Args:
        data (np.ndarray) : data to convert
    Returns:
        mfcc (np.ndarray) : 2d array of mfcc
    """
    mfcc = librosa.feature.mfcc(y=data)
    return mfcc


def convert_mel_spectrum_to_wav(data: np.ndarray):
    """Converst mel spectrum to wav
    Args:
        data (np.ndarray) : data to convert
    Returns:
        wav (np.ndarray) : 1d array of wav
    """
    wav = librosa.feature.inverse.mel_to_audio(data)
    return wav

def convert_mfcc_to_wav(data: np.ndarray):
    """Converst mfcc to wav
    Args:
        data (np.ndarray) : data to convert
    Returns:
        wav (np.ndarray) : 1d array of wav
    """
    wav = librosa.feature.inverse.mfcc_to_audio(data)
    return wav


if __name__ == "__main__":
    test_data = np.random.rand(22050)
    print("test data shape", test_data.shape)

    mel_spectrum = convert_to_mel_spectrum(test_data)
    print("mel spectrum shape", mel_spectrum.shape)
