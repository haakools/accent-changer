import os 
import sys
import numpy as np
from glob import glob

dataset_path = os.path.join(os.getcwd(), "dataset")

def read_txt_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return lines

labels = read_txt_file(os.path.join(dataset_path, "harvard_sentences.txt"))

# Need to refine the labels list to nicely format
