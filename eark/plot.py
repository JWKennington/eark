"""Plotting utilities for eark

"""
import types

import matplotlib.pyplot as plt
from palettable.scientific.sequential import Batlow_6 as cmap

from eark import solver

DENSITY_COLORS = cmap.mpl_colors


def plot_solution(soln: solver.Solution, neutron_color: str = 'red', show_densities: bool = True, output_file: str = None, legend_position: str = 'upper left',
                  y_transform: types.FunctionType = None):
    """Plot a solution

    Args:
        soln:
            Solution, the solution object from the inhour.solve function
        neutron_color:
            str, default 'red', the color to plot the neutron line
        show_densities:
            bool, default True, if True plot the precursor densities on a separate y axis
        output_file:
            str, default None, if specified, output the image file to this location instead of showing
        legend_position:
            str, default 'upper left', the location of the legend
        y_transform:
            Function, default None, if specified use this function on the y-axis. Examples are numpy.log, numpy.abs, etc

    Returns:
        Plot
    """
    # Plot neutron population
    y_transform_name = '' if y_transform is None else ' (' + y_transform.__name__ + ')'
    t = soln.t

    fig, ax1 = plt.subplots()
    lines = []

    # Population plot
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Neutron Population{}'.format(y_transform_name), color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    lines = ax1.plot(t, y_transform(soln.neutron_population) if y_transform is not None else soln.neutron_population, color=neutron_color, label='n') + lines

    if show_densities:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Precursor Densities{}'.format(y_transform_name), color='black')  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor='black')

        for i in range(1, soln.num_densities + 1):  # 1-indexed to match the math
            lines.extend(ax2.plot(t, y_transform(soln.precursor_density(i)) if y_transform is not None else soln.precursor_density(i), color=DENSITY_COLORS[i - 1],
                                  label='c_{:d}'.format(i)))

    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc=legend_position)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)


def plot_power(soln: solver.Solution, neutron_color: str = 'red', legend_position: str = 'upper left'):
    t = soln.t
    power = soln.neutron_population
    plt.plot(t, power, color=neutron_color, label='$P(t)$', marker='.')
    plt.xlabel("Time [s]")
    plt.ylabel("Power")
    plt.title("Power vs. Time")
    plt.legend()
    plt.show()


def plot_precursordensities(soln: solver.Solution, color: str = 'red', legend_position: str = 'upper left'):
    t = soln.t
    for i in range(1, soln.num_densities + 1):
        plt.plot(t, soln.precursor_density(i), color=DENSITY_COLORS[i - 1], marker='.',
                 label='$c_{:d}$'.format(i))
    plt.xlabel('Time $[s]$')
    plt.ylabel("Concentration of Neutron Precursors, $c_i [\#/dr^3]$")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(13, 13), useMathText=True)
    plt.title("Concentration of Neutron Precursors vs. Time$")
    plt.legend()
    plt.show()


def plot_T_mod(soln: solver.Solution, color: str = 'red', legend_position: str = 'upper left'):
    t = soln.t
    T_mod = soln.T_mod
    plt.plot(t, T_mod, color=color, label='$T_{mod}$', marker='.')
    plt.xlabel("Time [s]")
    plt.ylabel("Moderator Temperature [K]")
    plt.title("Moderator Temperature vs. Time")
    plt.legend()
    plt.show()

def plot_T_fuel(soln: solver.Solution, color: str = 'red', legend_position: str = 'upper left'):
        t = soln.t
        T_fuel = soln.T_fuel
        plt.plot(t, T_fuel, color=color, label='$T_{fuel}$', marker='.')
        plt.xlabel("Time [s]")
        plt.ylabel("Fuel Temperature [K]")
        plt.title("Fuel Temperature vs. Time")
        plt.legend()
        plt.show()

def plot_rho_temp(soln: solver.Solution, color: str = 'red', legend_position: str = 'upper left'):
    t = soln.t
    rho_temp = soln.rho_temp
    plt.plot(t, rho_temp, color=color, label='$\\rho_{temp}$', marker='.')
    plt.xlabel("Time [s]")
    plt.ylabel("Reactivity Temperature [$\Delta k/K$]")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-4, -4), useMathText=True)
    plt.title("Reactivity Temperature vs. Time")
    plt.legend()
    plt.show()

def plot_theta_c(soln: solver.Solution, color: str = 'red', legend_position: str = 'upper left'):
    t = soln.t
    theta_c = soln.theta_c
    plt.plot(t, theta_c, color=color, label='$\\theta_{CD}$', marker='.')
    plt.xlabel("Time [s]")
    plt.ylabel("Control Drum Angle $\\theta_{CD}$ [Degrees]")
    plt.title("Control Drum Angle vs. Time")
    plt.legend()
    plt.show()

def plot_rho_con(soln: solver.Solution, color: str = 'red', legend_position: str = 'upper left'):
    t = soln.t
    rho_con = soln.rho_con
    plt.plot(t, rho_con, color=color, label='$\\rho_{ext}$', marker='.')
    plt.xlabel("Time [s]")
    plt.ylabel("Control Drum Reactivity [$\Delta k$]")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-4, -4), useMathText=True)
    plt.title("Control Drum Reactivity vs. Time")
    plt.legend()
    plt.show()

def plot_angle_rho_con(soln: solver.Solution, color: str = 'red', legend_position: str = 'upper left'):
    theta_c = soln.theta_c
    rho_con = soln.rho_con
    plt.plot(theta_c, rho_con, color=color, label='$\\rho_{ext}$', marker='.')
    plt.xlabel("Drum Angle [degrees]")
    plt.ylabel("Control Drum Reactivity [$\Delta k$]")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-4, -4), useMathText=True)
    plt.title("Control Drum Angle vs. Control Drum Reactivity")
    plt.legend()
    plt.show()

