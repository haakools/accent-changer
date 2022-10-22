import os
from re import I
import sys
from glob import glob
import numpy as np

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
        "telugu": "dataset/data/indian/speaker_02"
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
        print(" --- Loading data --- ")
        raw_data = [load_wav_file_from_path(file_path) for file_path in file_paths]
        print(" --- Data loaded. ---")

        # Pad the raw_data to the max length
        sample_lengths = [len(data) for data in raw_data]

        # Get the max length
        assert len(sample_lengths) > 0, f"{speaker_path} has no data.\
            sample_lengths : {sample_lengths}"
        max_length = max(sample_lengths)

        print(" --- Padding data --- ")
        padded_data = [pad_end_of_data(data, max_length) for data in raw_data]
        print(" --- Data padded --- ")


        # Convert to mel spectrum   
        print(" --- Converting to mel spectrum --- ")
        mel_spectrums = [convert_to_mel_spectrum(data) for data in padded_data] 
        print(" --- Data converted to mel spectrum --- ")


        # Combine the mel_spectrums to a single array
        mel_spectrums = np.array(mel_spectrums)

        # Save the mel_spectrums to a npz file
        print(" --- Saving data --- ")
        target_folder = os.path.join("preprocessed_data", "mel_spectrums", gender)
        save_npz_to_path(mel_spectrums, target_folder, accent)
        print(" --- Data saved --- ")