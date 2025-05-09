import os
from typing import List
import matplotlib.pyplot as plt


def plot(data: List, label: str, xlabel: str, ylabel: str, title: str, save_path: str):
    """
    Plots a line graph based on the provided data and saves it to the specified path.

    Args:
        data (List): A list of numerical values to be plotted on the y-axis.
        label (List): A list of labels for the data series.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        save_path (str): File path where the plot will be saved.

    Returns:
        None
    """
    plt.figure()

    # min_value = min(data)
    X = range(1, len(data) + 1)
    plt.plot(X, data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
