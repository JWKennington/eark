"""Parameters for unittests, borrowed from scripts/run on 20200506"""


import numpy as np


################## PHYSICS PARAMETERS #############
N_INITIAL = 100                                               # initial Reactor Power                    [W]
BETA = 0.0071                                                   # delayed neutron fraction
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
PRECURSOR_DENSITY_INITIAL = BETA_VECTOR / (PRECURSOR_CONSTANTS * PERIOD) * N_INITIAL
RHO_FUEL_TEMP0 = BETA * 0.763


################## TH PARAMETERS ##################
C_F = 200                                                     # specific Heat Capacity of Fuel           [J/kg/K]
C_M = 4000                                                    # specific Heat Capacity of Moderator      [J/kg/K]
h = 4e6                                                       # heat transfer coefficient fuel/moderator [J/K/sec]
M_F = 575                                                     # mass of Fuel                             [kg]
M_M = 1000                                                    # mass of Moderator                        [kg]
W_M = 22                                                      # total moderator/coolant mass flow rate   [kg/sec]
T_in = 300                                                    # inlet coolant temperature                [K]
T_mod0 = 700                                                  # initial moderator temperature            [K]
T_fuel0 = 2000                                                # initial fuel temperature                 [K]
FUEL_GAS_DENSITY = 0.001                                      # fuel element gas density                 [g/cc]
MODR_GAS_DENSITY = 0.015                                      # moderator return channel gas density     [g/cc]
MODS_GAS_DENSITY = 0.035                                      # moderator supply channel gas density     [g/cc]


################# FUEL PIN PARAMETERS ##############
L_F       = 75                                                 # Length of Fuel Element                   [cm]
D_FLAT    = 1.9050                                             # Flat-to-Flat distance of Fuel Element    [cm]
D_COOLANT = 0.3454                                             # Coolant Channel Diameter                 [cm]
D_EFF = D_FLAT * ((2 * np.sqrt(3)) / np.pi)**0.5               # Effective Equivalent Unit Cell Diameter  [cm]
A_H = np.pi * L_F * D_COOLANT                                  # Heat Interface Area                      [cm^2]
V_F = np.pi * (D_EFF**2 - D_COOLANT**2) * L_F                  # Fuel Material Volume                     [cm^3]

########### CONTROL DRUM PARAMETERS ################
CDSPD   =  -1.0                                                 # control drum rotation speed             [deg/sec]
THETA_C0 = 100                                                 # initial angle of control drum           [deg]
