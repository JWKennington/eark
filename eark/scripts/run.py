import numpy as np

from eark import solver, plot

###################################################
#         USER INPUTS - DEFINE QUANTITIES         #
###################################################
a_F = -1.0e-5  # coefficient of reactivity for fuel              [dk/k]
a_M = -5.0e-5  # coefficient of reactivity for moderator         [dk/K]
BETA = 0.0075  # delayed neutron fraction
BETA_VECTOR = np.array([0.00247, 0.0013845, 0.001222, 0.0026455, 0.000832, 0.000169])
C_F = 200  # specific Heat Capacity of Fuel [J/kg/K]
C_M = 4000  # specific Heat Capacity of Moderator             [J/kg/K]

h = 4e6  # heat transfer coefficient of fuel and moderator [J/K/sec]
M_F = 40000  # mass of Fuel                                    [kg]
M_M = 7000  # mass of Moderator                               [kg]
N_INITIAL = 1600e6  # initial Reactor Power [W]
PERIOD = 0.001  # effective generation time [s]
PRECURSOR_CONSTANTS = np.array([0.0124, 0.0305, 0.1110, 0.3011, 1.1400, 3.0100])
PRECURSOR_DENSITY_INITIAL = np.array([298790322580, 68090163934, 16513513513, 13179176353, 1094736842, 84219269])
W_M = 8000  # total moderator/coolant mass flow rate          [kg/sec]
RHO = 0.1 * BETA
T_in = 550  # inlet coolant temperature [K]


def main():
    # Setup
    T_mod0 = T_in + (N_INITIAL / (2 * W_M * C_M))  # initial moderator temperature [K]
    T_fuel0 = T_in + (1 / (2 * W_M * C_M) + (1 / h)) * N_INITIAL  # initial fuel temperature  [K]

    # Solve
    soln = solver.solve(n_initial=N_INITIAL, precursor_density_initial=PRECURSOR_DENSITY_INITIAL,
                        beta_vector=BETA_VECTOR, precursor_constants=PRECURSOR_CONSTANTS,
                        rho=RHO, total_beta=BETA, period=PERIOD, h=h, M_M=M_M,
                        C_M=C_M, W_M=W_M, M_F=M_F, C_F=C_F, T_in=T_in,
                        T_mod0=T_mod0, T_fuel0=T_fuel0, t_max=10, num_iters=10000)

    plot.plot_power(soln)
    plot.plot_precursordensities(soln)
    plot.plot_T_mod(soln)
    plot.plot_T_fuel(soln)


if __name__ == '__main__':
    main()
