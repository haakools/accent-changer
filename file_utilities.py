import json
import scipy.io.wavfile as wav
import numpy as np


# File Utilities


def save_dict_to_json(d, json_path):
    """Saves dict of data to json file
    Args:
        d: (dict) of data
        json_path: (string) of json path
    """
    with open(json_path, 'w') as f:
        d = json.dump(d, f, indent=4)



def convert_wav_to_npz(wav_path, npz_path):
    rate, data = wav.read(wav_path)
    np.savez(npz_path, data=data)
    return rate, data