from glob import glob
import os
import json
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt


from file_utilities import save_dict_to_json

data_path = os.path.join("dataset", "accentdb_extended", "data")
accent_folder_path = glob(os.path.join(data_path,"*"))
accent_types = [accent.split(os.sep)[-1] for accent in accent_folder_path]

print("Avaiable accent types:", accent_types)
# Create metadata count of all the available accent types
metadata_count = {}
for accent in accent_types:
    accent_paths = glob(os.path.join(data_path, accent, "*"))
    metadata_count[accent] = len(accent_paths)

save_dict_to_json(metadata_count, os.path.join("metadata_count.json"))


# Try to load one file and do the preprocessing on it
import librosa

test_data_path = glob(os.path.join(data_path, "american", "speaker_01", "*.wav"))[0]
def convert_wav_to_mel_spectrum(wav_path):
    """Loads .wav file from path and converts to melspectrum
    Args;
        wav_path (str): path to wav file
    Returns:
        spectrum (np.ndarray) : 2d array of spectrum    
    """

    signal, rate = librosa.load(test_data_path)
    spectrum = librosa.feature.melspectrogram(y=signal, sr=rate)
    spectrum = np.log10(spectrum)
    return spectrum

def save_spectrum_to_npz(spectrum):

    

