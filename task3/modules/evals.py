#!/usr/bin/env python3

import matplotlib.pyplot as plt

def plot_lossvsepoch(epochs, acc, filename):
    plt.ioff()  # Set mode on: non-interactive

    fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        ax.scatter(epochs, acc, "o")
        ax.savefig(filename)

