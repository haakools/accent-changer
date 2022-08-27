from glob import glob
import os
import json
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import librosa
from typing import List

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




def convert_wav_to_mel_spectrum(wav_path):
    """Loads .wav file from path and converts to melspectrum
    Args;
        wav_path (str): path to wav file
    Returns:
        spectrum (np.ndarray) : 2d array of spectrum    
    """

    signal, rate = librosa.load(wav_path)
    
    spectrum = librosa.feature.melspectrogram(y=signal, sr=rate)
    spectrum = np.log10(spectrum+1e-6) #adding small fraction to avoid log of zero
    return spectrum


def save_spectrum_to_npz(spectrum: List[np.ndarray]) -> None:

    """
    Saves spectrum to npz file
    Args:
        spectrum (np.ndarray): 2d array of spectrum
    """




    np.savez("test_spectrum.npz", data)
# https://github.com/b2slab/padding_benchmark/blob/master/src/preprocessing.pys


test_data_path = glob(os.path.join(data_path, "american", "speaker_01", "*.wav"))
data = []
i=0
for entry in test_data_path:
    data.append(convert_wav_to_mel_spectrum(entry))
    i +=1
    if i== 3:
        break

from keras.preprocessing.sequence import pad_sequences
padded_data = pad_sequences(data)

plt.subplot(211)
plt.imshow(data[0])
plt.subplot(212)
plt.imshow(padded_data[0])
plt.show()
exit()
save_spectrum_to_npz(data)
