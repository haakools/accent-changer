import numpy as np



def pad_end_of_data(data: np.ndarray, max_length: int):
    """Pads the end of the data with zeros
    Args:
        data (np.ndarray) : data to pad
        max_length (int) : length to pad the data to

    Returns:
        padded_data (np.ndarray) : padded data
    """
    pad_before = 0
    pad_after = max_length - data.shape[0]
    padded_data = np.pad(data, (pad_before, pad_after), 'constant', constant_values=0.0)

    return padded_data