import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

repoDir = os.path.abspath(Path(__file__).parent.parent)
sys.path.append(repoDir)
from utils.file_utilities import load_npz_from_path 

if __name__ == "__main__":
    path_american = os.path.join(repoDir, "preprocessed_data", "mel_spectrums", "female", "american.npz")
    path_bangla = os.path.join(repoDir, "preprocessed_data", "mel_spectrums", "female", "bangla.npz")

    american_data = load_npz_from_path(path_american)
    bangla_data = load_npz_from_path(path_bangla)

    print(american_data.shape)
    print(bangla_data.shape)


    for sample in bangla_data:
        # Calculate the mean of the sample for each timestep
        mean = np.mean(np.exp(sample), axis=0)
        plt.plot(mean)
    plt.title("Bangla")
    plt.show()

    print("Bangla mean: ", np.mean(mean))
    for sample in american_data:
        # Calculate the mean of the sample for each timestep
        mean = np.mean(np.exp(sample), axis=0)
        plt.plot(mean)
    plt.title("American")
    plt.show()

    print("Bangla mean: ", np.mean(mean))


    plt.subplot(2, 1, 1)
    plt.imshow(american_data[0])
    plt.title("american data sample 0")
    plt.subplot(2, 1, 2)
    plt.imshow(bangla_data[0])
    plt.title("bangla data sample 0")
    plt.show()