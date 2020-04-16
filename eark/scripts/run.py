import argparse
import logging

import numpy as np

from eark import solver, plot

# User Input - these set defaults for script
DEFAULT_N_INITIAL = 1500e6  # initial Reactor Power                    [Watts]
DEFAULT_BETA = 0.0075  # delayed neutron fraction
DEFAULT_PERIOD = 0.001  # effective generation time
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--power', type=float, default=DEFAULT_N_INITIAL)
    parser.add_argument('-b', '--beta', type=float, default=DEFAULT_BETA)
    parser.add_argument('-l', '--period', type=float, default=DEFAULT_PERIOD)
    return parser.parse_args()


def main(args):
    n_initial, beta, period = args.power, args.beta, args.period
    logger.info('Running Eark Solver with:\n  N_initial = {:.2e}\n  Beta = {:.2e}\n  Period = {:.2e}'.format(n_initial, beta, period))

    beta_vector = np.array([0.00247, 0.0013845, 0.001222, 0.0026455, 0.000832, 0.000169])
    precursor_constants = np.array([0.0124, 0.0305, 0.1110, 0.3011, 1.1400, 3.0100])
    precursor_density_initial = ([298790322580, 68090163934, 16513513513, 13179176353, 1094736842, 84219269])
    rho = 0.1 * beta

    C_F = 200  # specific Heat Capacity of Fuel                  [J/kg/K]
    C_M = 4000  # specific Heat Capacity of Moderator             [J/kg/K]
    M_F = 40000  # mass of Fuel                                    [kg]
    M_M = 7000  # mass of Moderator                               [kg]
    h = 4e6  # heat transfer coefficient of fuel and moderator [J/K/sec]
    W_M = 8000  # total moderator/coolant mass flow rate          [kg/sec]

    a_F = -1.0e-5  # coefficient of reactivity for fuel              [dk/k]
    a_M = -5.0e-5  # coefficient of reactivity for moderator         [dk/K]

    T_in = 550  # inlet coolant temperature [K]
    T_mod0 = T_in + (n_initial / (2 * W_M * C_M))  # initial moderator temperature [K]
    T_fuel0 = T_in + (1 / (2 * W_M * C_M) + (1 / h)) * n_initial  # initial fuel temperature  [K]

    # Solve
    soln = solver.solve(n_initial=n_initial, precursor_density_initial=precursor_density_initial,
                        beta_vector=beta_vector, precursor_constants=precursor_constants,
                        rho=rho, total_beta=beta, period=period, h=h, M_M=M_M,
                        C_M=C_M, W_M=W_M, M_F=M_F, C_F=C_F, T_in=T_in,
                        T_mod0=T_mod0, T_fuel0=T_fuel0, t_max=10, num_iters=30000)

    plot.plot_power(soln)
    plot.plot_precursordensities(soln)
    plot.plot_T_mod(soln)


if __name__ == '__main__':
    args = parse_args()
    main(args)
