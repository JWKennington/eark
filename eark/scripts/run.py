import numpy as np

from eark import solver
from eark.control import LinearControlRule

###################################################
#         USER INPUTS - DEFINE QUANTITIES         #
###################################################

################## PHYSICS PARAMETERS #############


POWER_INITIAL = 25e6                                          # initial Reactor Power                    [W]
BETA = 0.0071                                                 # delayed neutron fraction
BETA_VECTOR = np.array([2.23985e-4,
                        1.18115e-3,
                        1.16108e-3,
                        3.29914e-3,
                        1.00849e-3,
                        3.57418e-4])
PERIOD = 2.63382e-5                                           # effective generation time                [s]
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
MASS_FLOW = 22                                                # total moderator/coolant mass flow rate   [kg/sec]
TEMP_IN = 300                                                 # inlet coolant temperature                [K]
TEMP_MOD_INITIAL = TEMP_IN + \
                   (POWER_INITIAL / (2 * MASS_FLOW * HEAT_CAP_MOD))

TEMP_FUEL_INITIAL = TEMP_IN + \
                    (1 / (2 * MASS_FLOW * HEAT_CAP_MOD) + (1 / HEAT_COEFF)) * POWER_INITIAL

FUEL_GAS_DENSITY = 0.001                                      # fuel element gas density                 [g/cc]
MODR_GAS_DENSITY = 0.015                                      # moderator return channel gas density     [g/cc]
MODS_GAS_DENSITY = 0.035                                      # moderator supply channel gas density     [g/cc]


################# FUEL PIN PARAMETERS ##############           (NOT IN USE CURRENTLY!)
L_F       = 75                                                 # Length of Fuel Element                   [cm]
D_FLAT    = 1.9050                                             # Flat-to-Flat distance of Fuel Element    [cm]
D_COOLANT = 0.3454                                             # Coolant Channel Diameter                 [cm]
D_EFF = D_FLAT * ((2 * np.sqrt(3)) / np.pi)**0.5               # Effective Equivalent Unit Cell Diameter  [cm]
A_H = np.pi * L_F * D_COOLANT                                  # Heat Interface Area                      [cm^2]
V_F = np.pi * (D_EFF**2 - D_COOLANT**2) * L_F                  # Fuel Material Volume                     [cm^3]

########### CONTROL DRUM PARAMETERS ################
DRUM_SPEED   =  LinearControlRule(coeff=0, const= 0.0, t_min=0, t_max=0)

DRUM_ANGLE_INITIAL = 64.65                                     # initial angle of control drum            [deg]

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
                        drum_angle_initial= DRUM_ANGLE_INITIAL,
                        t_max= 100,
                        num_iters=1000)

    # Plot
    soln.plot_power()
    soln.plot_densities()
    soln.plot_temp_mod()
    soln.plot_temp_fuel()
    soln.plot_rho_fuel_temp()
    soln.plot_rho_mod_temp()
    soln.plot_rho_con_drum()
    soln.plot_rho_con_drum_angle()




if __name__ == '__main__':
    main()
