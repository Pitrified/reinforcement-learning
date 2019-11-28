import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from timeit import default_timer as timer


def plot_scores(scores, fig_num=1, y_label=None, clear_plot=True):
    """Plot scores of training

    fig_num: fig to plot to
    y_label: label of the y axis
    clear_plot: wether to clear the fig or add the info on it
    """

    if y_label is None:
        y_label = "Score"

    plt.figure(fig_num)

    if clear_plot:
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel(y_label)

    plt.plot(scores)

    plt.pause(0.001)  # pause a bit so that plots are updated
