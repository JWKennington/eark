import numpy as np

from eark import solver
from eark.control import LinearControlRule

###################################################
#         USER INPUTS - DEFINE QUANTITIES         #
###################################################

################## PHYSICS PARAMETERS #############


POWER_INITIAL = 25e6                                          # initial Reactor Power                    [W]
BETA = 7.23126E-03                                            # delayed neutron fraction
BETA_VECTOR = np.array([2.23985e-4,
                        1.18115e-3,
                        1.16108e-3,
                        3.29914e-3,
                        1.00849e-3,
                        3.57418e-4])
PERIOD = 2.63382E-05                                            # effective generation time                [s]
PRECURSOR_CONSTANTS = np.array([1.24906e-2,
                                3.17621e-2,
                                1.09665e-1,
                                3.18385e-1,
                                1.35073e0,
                                8.73657e0])
PRECURSOR_DENSITY_INITIAL = BETA_VECTOR / (PRECURSOR_CONSTANTS * PERIOD) * POWER_INITIAL


################## TH PARAMETERS ##################
HEAT_CAP_FUEL = 200                                           # specific Heat Capacity of Fuel           [J/kg/K]
HEAT_CAP_MOD = 4000                                           # specific Heat Capacity of Moderator      [J/kg/K]
HEAT_COEFF = 4e6                                              # heat transfer coefficient fuel/moderator [J/K/sec]
MASS_FUEL = 575                                               # mass of Fuel                             [kg]
MASS_MOD = 1000                                               # mass of Moderator                        [kg]
MASS_FLOW = 14                                                # total moderator/coolant mass flow rate   [kg/sec]
TEMP_IN = 350                                                 # inlet coolant temperature                [K]
TEMP_MOD_INITIAL = TEMP_IN + \
                   (POWER_INITIAL / (2 * MASS_FLOW * HEAT_CAP_MOD))

TEMP_FUEL_INITIAL = TEMP_IN + \
                    (1 / (2 * MASS_FLOW * HEAT_CAP_MOD) + (1 / HEAT_COEFF)) * POWER_INITIAL



########### CONTROL DRUM PARAMETERS ################
DRUM_SPEED   =  LinearControlRule(coeff=0, const= 12.0, t_min=20, t_max=30)


def main():

    # Solve
    soln = solver.solve(power_initial=POWER_INITIAL,
                        precursor_density_initial=PRECURSOR_DENSITY_INITIAL,
                        beta_vector=BETA_VECTOR,
                        precursor_constants=PRECURSOR_CONSTANTS,
                        total_beta=BETA, period=PERIOD,
                        heat_coeff=HEAT_COEFF,
                        mass_mod=MASS_MOD,
                        heat_cap_mod=HEAT_CAP_MOD,
                        mass_flow=MASS_FLOW,
                        mass_fuel=MASS_FUEL,
                        heat_cap_fuel=HEAT_CAP_FUEL,
                        temp_in=TEMP_IN,
                        temp_mod_initial=TEMP_MOD_INITIAL,
                        temp_fuel_initial=TEMP_FUEL_INITIAL,
                        drum_control_rule=DRUM_SPEED,
                        t_max= 500,
                        num_iters=10000)

    # Plot
    soln.plot_power()
    soln.plot_densities()
    soln.plot_temp_fuel()
    soln.plot_temp_mod()
    soln.plot_rho_fuel_temp()
    soln.plot_rho_mod_temp()
    soln.plot_rho_con_drum()
    soln.plot_rho_total()
    soln.plot_rho_con_drum_angle()





if __name__ == '__main__':
    main()
