"""Plotting utilities for eark

"""

import typing

import matplotlib.pyplot as plt
from palettable.scientific.sequential import Batlow_6 as cmap

DENSITY_COLORS = cmap.mpl_colors


def plot_soln_quantity(t, y, y_label: str = 'Solution', title: str = 'Solution Plot', x_label: str = 'Time (s)', color: str = 'red', label: typing.Callable = None,
                       marker: str = '.', output_file: str = None):
    if not isinstance(y, list):
        y = [y]
        color = [color]

    for i in range(len(y)):
        label = label if isinstance(label, str) else label(i) if callable(label) else y_label
        plt.plot(t, y[i], color=color[i], label=label, marker=marker)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
