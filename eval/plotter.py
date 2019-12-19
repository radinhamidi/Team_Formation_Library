import matplotlib.pyplot as plt
import numpy as np


def plot_at_k(k_set, values, label, title='', save_to_file=False, file_name='Untitled.png', dir='./'):
    # Extracting data from json files of records
    y = [values[k] for k in k_set]

    plt.figure(0)
    plt.xlabel = 'K'
    plt.ylabel = label
    plt.title = title
    plt.plot(k_set, y, label=label)
    plt.legend(loc='best')
    plt.grid()

    if save_to_file:
        plt.savefig(dir + file_name)

    plt.show()
