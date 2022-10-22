import json
import numpy as np
import os
import librosa
from typing import List
import tqdm
"""Functions to save and load files"""






def load_multiple_wav_files_from_path(wav_path: List[str]) -> List[np.ndarray]:
    """Loads multiple wav files from path
    Args:
        wav_path (List[str]) : list of paths to wav files
    Returns:
        wav_data (List[np.ndarray]) : list of wav data
    """
    # add tqmd load bar here to keep track of progress
    wav_data = []
    for path in tqdm.tqdm(wav_path):
        data = load_wav_file_from_path(path)
        wav_data.append(data)
    return wav_data


def load_wav_file_from_path(wav_path):
    """Loads wav file from path
    Args:
        wav_path (string) : path to wav file
    Returns:
        data (np.ndarray) : 1d array of wav data
    """
    # Add assertion to check if file exists
    assert os.path.exists(wav_path), f"File {wav_path} does not exist"
 
    # Loading the wavfile with librosa for compatiblitiy with mel spectrogram 
    desired_sample_rate = 22050
    data, sample_rate = librosa.load(wav_path, sr=desired_sample_rate)
    
    # Asssertion to check that the sampling rate is similiar for all data
    assert sample_rate == desired_sample_rate, f"Sampling rate of {sample_rate} is \
        not {desired_sample_rate}"

    # TODO: Add assertion for number of channels to the data
    data = np.squeeze(data)
    return data

def load_npz_from_path(npz_path):
    """Loads npz file and returns data
    Args:
        npz_path (str) : path to npz file
    Returns:
        data (np.ndarray) : data from npz file
    """
    data = np.load(npz_path)['data']
    return data

def save_npz_to_path(data, save_to_path, file_name):
    """Saves data to npz file
    Args:
        data (np.ndarray) : data to save
        npz_path (str) : path to save data
    """
    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)

    np.savez(os.path.join(save_to_path, file_name), data=data)

def load_json_to_dict(json_path):
    """Loads json file to dict
    Args:
        json_path: (string) of json path
    Returns:
        d: (dict) of data
    """
    with open(json_path, 'r') as f:
        d = json.load(f)
    return d

def save_dict_to_json(d, json_path):
    """Saves dict of data to json file
    Args:
        d: (dict) of data
        json_path: (string) of json path
    """
    with open(json_path, 'w') as f:
        d = json.dump(d, f, indent=4)
