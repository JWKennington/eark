# Emulating Astronautical Reactor Kinetics
The `eark` module is a compilation of utilities for analyzing reactor kinetics and transients for specific 
Nuclear Thermal Propulsion (NTPs) rockets.

[![Build Status](https://travis-ci.com/vigneshwar-manickam/eark.svg?branch=master)](https://travis-ci.com/vigneshwar-manickam/eark) 

## Installation
The `eark` package is pip-installable and is tested against python 3.6/3.7 in both a Linux and Mac OSX environment.

```bash
pip install eark
conda env create -f environment.yml
```  

## Using `eark`

### Run File
The `eark.run` module contains utilities for solving the Points Kinetics Equations based on user-imported initial 
conditions, for example:
```python
import numpy as np

from eark import solver

###################################################
#         USER INPUTS - DEFINE QUANTITIES         #
###################################################

################## PHYSICS PARAMETERS #############
from eark.control import LinearControlRule

POWER_INITIAL = 25e6                                            # initial Reactor Power                    [W]
BETA = 0.0071                                                   # delayed neutron fraction
BETA_VECTOR = np.array([2.23985e-4,
                        1.18115e-3,
                        1.16108e-3,
                        3.29914e-3,
                        1.00849e-3,
                        3.57418e-4])
PERIOD = 2.63382e-5                                             # effective generation time                [s]
PRECURSOR_CONSTANTS = np.array([1.24906e-2,
                                3.17621e-2,
                                1.09665e-1,
                                3.18385e-1,
                                1.35073e0,
                                8.73657e0])
PRECURSOR_DENSITY_INITIAL = BETA_VECTOR / (PRECURSOR_CONSTANTS * PERIOD) * POWER_INITIAL
RHO_FUEL_TEMP_INITIAL = BETA * -0.671645
RHO_MOD_TEMP_INITIAL = BETA * -0.0204816

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


```
### Unique Drum Rotation Functionality
 `eark` is currently able to model control drum rotation speed as a linear function. In the future, we hope to implement control logic laws
 that allows the Drum Rotation to freely rotate to maintain temperature during the startup sequence of NTPs. The `eark.run` module allows the user
 to define the drum speed as a linear function by the LinearControlRule. User's are able to start the ramp of reactivity insertion at `t_min` and finish
 the reactivity insertion at `t_max.` A constant speed can be used by entering an integer value the `const` argument or a time-varying speed can be implemented 
 by entering an integer value into the `coeff` argument. 
 
 ```python
########### CONTROL DRUM PARAMETERS ################
DRUM_SPEED   =  LinearControlRule(coeff=0, const=2, t_min=20, t_max=30)
DRUM_ANGLE_INITIAL = 70                                      # initial angle of control drum           [deg]
```

As shown above, the control drums will rotate at 2 deg/sec starting at 20 seconds for 10 seconds with an initial 
critical angle of 70 degrees. The drums will have rotated to 90 degrees by the end of the transient at 30 seconds. 

### Running the test suite
The simplest usage of `eark` is to run the test suite. This can ensure the installation was successful.
```python
>>> import eark
>>> eark.run_tests()
```
