"""Module for defining the time-derivatives of the reactor state and the dynamical interactions between coupled variables.

References:
    [1] Witter JK. Modeling for the Simulation and Control of Nuclear Rocket Systems [Ph.D.]. [Department of Nuclear Engineering]: Massachusetts Institute of Technology; 1993.
"""

import numpy as np


#################################################
#             POPULATION DYNAMICS               #
#################################################

def total_neutron_deriv(beta: float, period: float, n, precursor_constants: np.ndarray,
                        precursor_density: np.ndarray, rho_fuel_temp: float, T_mod: float, fuel_gas_density: float,
                        modr_gas_density: float, mods_gas_density: float, theta_c: float) -> float:
    """Compute time derivative of total neutron population (i.e. reactor power), $\frac{dn}{dt}(t)$

    Args:

        beta:
            float, delayed neutron fraction  []
        period:
            float, effective generation time [seconds]
        n:
            float, reactor power [W]
        precursor_constants:
            ndarray, 1x6 array of lambda_i
        precursor_density:
            ndarray, 1x6 array of c_i
        rho_fuel_temp
            float, reactivity due to fuel temperature                    [dK]
        fuel_gas_density:
            float, gas density in the fuel                               [g/cc]
        modr_gas_density:
            float, gas density in return channel of moderator            [g/cc]
        mods_gas_density:
            float, gas density in supply channel of moderator            [g/cc]
        theta_c:
                float, angle of control drunk rotation                   [degrees]

    Returns:
        float, the time derivative of total neutron population or reactor power
    """
    total_rho = rho_fuel_temp + \
                mod_temp_reactivity(beta, T_mod) + \
                drum_reactivity(beta, theta_c)

    return (((total_rho - beta) / period) * n) + np.inner(precursor_constants, precursor_density)


def delay_neutron_deriv(beta_vector: np.ndarray, period: float, n: float, precursor_constants: np.ndarray,
                        precursor_density: np.ndarray) -> np.ndarray:
    """Compute time derivative of delayed neutron population, $\frac{dc_i}{dt}(t)$

    Args:
        beta_vector:
            ndarray, 1x6 vector of fraction of delayed neutrons of ith kind
        period:
            float, effective generation time [seconds]
        n:
            float, reactor power [W]
        precursor_constants:
            ndarray, 1x6 array of lambda_i
        precursor_density:
            ndarray, 1x6 array of c_i

    Returns:
        ndarray 1x6 vector of the time derivative of each of the "i" components of precursor density

    """
    return beta_vector * n / period - precursor_constants * precursor_density


#################################################
#              THERMAL DYNAMICS                 #
#################################################


def mod_temp_deriv(h: float, M_M: float, C_M: float, W_M: float, T_fuel: float, T_mod: float, T_in: float) -> float:
    """Compute time derivative of moderator temperature, $\frac{dT_mod}{dt}(t)$

    Args:
        h:
            float, heat transfer coefficient of fuel and moderator [J/K/sec]
        M_M:
            float, mass of moderator                               [kg]
        C_M:
            float, specific Heat capacity of moderator             [J/kg/K]
        W_M:
            float, total moderator/coolant mass flow rate          [kg/sec]
        T_fuel:
            float, temperature of fuel                             [K]
        T_mod:
            float, temperature of moderator                        [K]
        T_in:
            float, temperature of inlet coolant                    [K]
    """
    return (h / (M_M * C_M)) * (T_fuel - T_mod) - (2 * W_M / M_M) * (T_mod - T_in)


def fuel_temp_deriv(n: float, M_F: float, C_F: float, h: float, T_fuel: float, T_mod: float) -> float:
    """Compute time derivative of fuel temperature, $\frac{dT_fuel}{dt}(t)$

    Args:
        n:
            float, Reactor Power                                   [W]
        M_F:
            float, mass of fuel                                    [kg]
        C_F:
            float, specific heat capacity of fuel                  [J/kg/K]
        h:
            float, heat transfer coefficient of fuel and moderator [J/K/sec]
        T_fuel:
            float, temperature of fuel                             [K]
        T_mod:
            float, temperature of moderator                        [K]

    """
    return (n / (M_F * C_F)) - ((h / (M_F * C_F)) * (T_fuel - T_mod))


#################################################
#                DRUM DYNAMICS                  #
#################################################


def theta_c_deriv(cdspd: float) -> float:
    """
            Models rotation of drums, $\frac{dtheta_c}{dt}(t)$

            Args:
                cdspd:
                    float, rotation rate of control drums                       [degrees/sec]
    """
    return cdspd


#################################################
#                  REACTIVITY                   #
#################################################


def fuel_temp_reactivity_deriv(beta: float, T_fuel: float) -> float:
    """Compute time derivative of fuel temperature reactivity, $\frac{drho_fuel_temp}{dt}(t)$

    Args:
        beta:
            float, delayed neutron fraction                        []
        T_fuel:
            float, temperature of fuel                             [K]

    """

    return beta * (7.64e-7 * T_fuel - 3.36e-3)


def mod_temp_reactivity(beta: float, T_mod: float) -> float:
    """Compute reactivity due to moderator temperature.

                Args:
                    beta:
                         float, delayed neutron fraction                            []
                    T_mod:
                         float, temperature of moderator                            [K]

    """
    return beta * ((1.56e-7 * (T_mod) ** 2) - (1.70e-3 * (T_mod) + 0.666))


def drum_reactivity(beta: float, theta_c: float) -> float:
    """Compute reactivity due to control drum rotation

        Args:
            beta:
                float, delayed neutron fraction                          []
            theta_c:
                float, angle of control drunk rotation                   [degrees]

    """
    return beta * ((6.51e-6 * (theta_c) ** 3) - (1.76e-3 * (theta_c) ** 2) + (2.13e-2 * theta_c) + 4.92536)
