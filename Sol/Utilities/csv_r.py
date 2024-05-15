import os.path

import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    x = data['Step']
    z = data['Value']
    return x, z


# Function to plot data from two CSV files on the same plot
def plot_two_datasets(filepath1, filepath2):
    x1, z1 = load_and_prepare_data(filepath1)
    x2, z2 = load_and_prepare_data(filepath2)

    plt.figure(figsize=(10, 6))
    plt.scatter(x1, z1, c='blue',  label='Dataset 1: Value vs. Steps')
    plt.scatter(x2, z2, c='red', label='Dataset 2: Value vs. Steps')
    plt.title('Comparison of Value Number vs Steps from Two Datasets')
    plt.xlabel('Steps')
    plt.ylabel('Value Number')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_two_datasets(os.path.join("Sol/results/PPO_save_05.14.2024_00.00.34_PPO_1.csv"),
                  os.path.join("Sol/results/PPO_save_05.14.2024_01.46.39_PPO_1.csv"))
