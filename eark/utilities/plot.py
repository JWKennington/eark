"""Plotting utilities for eark

"""

import typing

import matplotlib.pyplot as plt
from palettable.scientific.sequential import Batlow_6 as cmap

DENSITY_COLORS = cmap.mpl_colors


def plot_soln_quantity(t, y, y_label: str = 'Solution', title: str = 'Solution Plot', x_label: str = '$t$ [s]',
                       color: str = 'red', label: typing.Union[typing.Callable, str] = None,
                       marker: str = '.', ymin: float = 0, ymax: float = 0,
                       sci: bool = False, scilimit: float = 0, output_file: str = None):
    if not isinstance(y, list):
        y = [y]
        color = [color]

    for i in range(len(y)):
        label_i = label if isinstance(label, str) else label(i) if callable(label) else y_label
        plt.plot(t, y[i], color=color[i], label=label_i, marker=marker)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(b=True, which='major', color='grey', linestyle='-', linewidth=0.4)
    if sci == True:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(scilimit,scilimit), useMathText=True)
    plt.ylim(ymin, ymax)
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.title(title)
    plt.legend()

    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
