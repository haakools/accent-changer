import tensorflow as tf
import os
import sys

sys.path.append("..")

from utils.file_utilities import load_npz_from_path

# Path to the data
path_american = os.path.join("..", "preprocessed_data", "mel_spectrums", "female", "american.npz")
path_bangla = os.path.join("..", "preprocessed_data", "mel_spectrums", "female", "bangla.npz")


# need to create normalizaiotn and train test splitting, which saves the output

def normalize_data(data):
    """Normalizes the data"""
    min_data
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def unnormalize_data(data):
    """Unnormalizes the data"""
    return data * (np.max(data) - np.min(data)) + np.min(data)



def prepare_dataset(path_x: str, path_y: str, buffer_size: int, batch_size: int) -> tf.data.Dataset:
    """Creates tensorflow dataset of data from path"""

    # Load the data
    x = load_npz_from_path(path_x)
    y = load_npz_from_path(path_y)
    
    # Min max normalize the numpy arrays
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Create the tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size).batch(batch_size) 
    return dataset
