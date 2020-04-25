import numpy as np

from eark import solver, plot

###################################################
#         USER INPUTS - DEFINE QUANTITIES         #
###################################################

################## PHYSICS PARAMETERS ###################
N_INITIAL = 1500e6                                            # initial Reactor Power                    [W]
a_F = -1.0e-5                                                 # coefficient of reactivity for fuel       [dk/k]
a_M = -5.0e-5                                                 # coefficient of reactivity for moderator  [dk/K]
BETA = 0.0075                                                 # delayed neutron fraction
BETA_VECTOR = np.array([0.00247,
                        0.0013845,
                        0.001222,
                        0.0026455,
                        0.000832,
                        0.000169])
PERIOD = 0.00001                                              # effective generation time                [s]
PRECURSOR_CONSTANTS = np.array([0.0124,
                                0.0305,
                                0.1110,
                                0.3011,
                                1.1400,
                                3.0100])
PRECURSOR_DENSITY_INITIAL = BETA_VECTOR / (PRECURSOR_CONSTANTS * PERIOD) * N_INITIAL
RHO_CON = 0.0                                                 # NEED TO WORK ON THIS. I THINK THIS IS EXCESS REACTIVITY

################## TH PARAMETERS ###################
C_F = 200                                                     # specific Heat Capacity of Fuel           [J/kg/K]
C_M = 4000                                                    # specific Heat Capacity of Moderator      [J/kg/K]
h = 4e6                                                       # heat transfer coefficient fuel/moderator [J/K/sec]
M_F = 40000                                                   # mass of Fuel                             [kg]
M_M = 7000                                                    # mass of Moderator                        [kg]
W_M = 8000                                                    # total moderator/coolant mass flow rate   [kg/sec]
T_in = 550                                                    # inlet coolant temperature                [K]
T_mod0 = T_in + (N_INITIAL / (2 * W_M * C_M))                 # initial moderator temperature            [K]
T_fuel0 = T_in + (1 / (2 * W_M * C_M) + (1 / h)) * N_INITIAL  # initial fuel temperature                 [K]

########### CONTROL DRUM PARAMETERS ###################
CDWRTH  = .0405 * BETA                                         # control drum worth
CDSPD   = 1.0                                                  # control drum rotation speed             [deg/sec]
THETA_C0 = 0.0                                                 # initial angle of control drum           [deg]

def main():

    # Solve
    soln = solver.solve(n_initial=N_INITIAL,
                        precursor_density_initial=PRECURSOR_DENSITY_INITIAL,
                        beta_vector=BETA_VECTOR,
                        precursor_constants=PRECURSOR_CONSTANTS,
                        total_beta=BETA, period=PERIOD,
                        h=h,
                        M_M=M_M,
                        C_M=C_M,
                        W_M=W_M,
                        M_F=M_F,
                        C_F=C_F,
                        T_in=T_in,
                        T_mod0=T_mod0,
                        T_fuel0=T_fuel0,
                        rho_con=RHO_CON,
                        a_F=a_F,
                        a_M=a_M,
                        cdspd=CDSPD,
                        cdwrth=CDWRTH,
                        theta_c0= THETA_C0,
                        t_max=90,
                        num_iters=1000)



    # Plot
    plot.plot_power(soln)
    plot.plot_precursordensities(soln)
    plot.plot_T_mod(soln)
    plot.plot_T_fuel(soln)
    plot.plot_rho_temp(soln)
    plot.plot_theta_c(soln)
    plot.plot_rho_con(soln)


if __name__ == '__main__':
    main()
