import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


sys.path.append("..")
from preprocessing.spectogram import convert_to_mel_spectrum
from preprocessing.padding import pad_end_of_data
from utils.file_utilities import load_npz_from_path, save_npz_to_path,\
    load_wav_file_from_path, save_dict_to_json


# Path to the data
speakers = {
    "female": {
        "american": "dataset/data/american/speaker_08",
        "bangla": "dataset/data/bangla/speaker_02",
    },
    "male": {
        "american": "dataset/data/american/speaker_07",
        "telugu": "dataset/data/indian/speaker_02",
    }
}

# Save the chosen speakers to a json file
save_dict_to_json(speakers, "preprocessed_data/speakers.json")

# Loop over both genders and speakers
for gender in speakers.keys():
    for accent in speakers[gender].keys():
        print(f"Processing {gender} {accent} data") 
        speaker_path = speakers[gender][accent]
        
        # Get all the file paths with glob
        file_paths = sorted(glob(os.path.join(speaker_path, "*.wav")))

        # Load all the file_paths and convert to mel spectrum
        raw_data = [load_wav_file_from_path(file_path) for file_path in file_paths]

        for i, data in enumerate(raw_data):
            print(data.shape)

        # Only keeping the first 100'000 data points
        n_datapoints = 100000 
        trimmed_data = [data[:n_datapoints] for data in raw_data]

        # Get the max length of the spectrums
        sample_lengths = [len(data) for data in trimmed_data]
        max_length = max(sample_lengths)
        padded_data = [pad_end_of_data(data, max_length) for data in trimmed_data]


        print(padded_data[0].shape) 
        # Convert to mel spectrum   
        mel_spectrums = [convert_to_mel_spectrum(data) for data in padded_data] 

        # printing some shapes :)
        print(len(mel_spectrums))
        print(mel_spectrums[0].shape)
        print(mel_spectrums[1].shape)

        # Combine the mel_spectrums to a single array
        output_array = np.stack(mel_spectrums)
        print("Output array shape: ", output_array.shape)

        # Save the mel_spectrums to a npz file
        target_folder = os.path.join("preprocessed_data", "mel_spectrums", gender)
        save_npz_to_path(output_array, target_folder, accent)