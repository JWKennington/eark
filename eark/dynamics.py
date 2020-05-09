"""Module for defining the time-derivatives of the reactor state and the dynamical interactions between coupled variables.

References:
    [1] Witter JK. Modeling for the Simulation and Control of Nuclear Rocket Systems [Ph.D.]. [Department of Nuclear Engineering]: Massachusetts Institute of Technology; 1993.
"""

import numpy as np


#################################################
#             POPULATION DYNAMICS               #
#################################################

def total_neutron_deriv(beta: float, period: float, power: float, precursor_constants: np.ndarray,
                        precursor_density: np.ndarray, rho_fuel_temp: float, rho_mod_temp: float, rho_con_drum:float) -> float:
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
            float, reactivity due to fuel temperature                    [dK/K]
        rho_mod_temp
            float, reactivity due to moderator temperature               [dK/K]
        rho_con_drum:
            float, reactivity due to control drum rotation               [dK/theta]

    Returns:
        float, the time derivative of total neutron population or reactor power
    """
    total_rho = rho_fuel_temp + \
                rho_mod_temp + \
                rho_con_drum

    return (((total_rho - beta) / period) * power) + np.inner(precursor_constants, precursor_density)


def delay_neutron_deriv(beta_vector: np.ndarray, period: float, power: float, precursor_constants: np.ndarray,
                        precursor_density: np.ndarray) -> np.ndarray:
    """Compute time derivative of delayed neutron population, $\frac{dc_i}{dt}(t)$

    Args:
        beta_vector:
            ndarray, 1x6 vector of fraction of delayed neutrons of ith kind
        period:
            float, effective generation time [seconds]
        power:
            float, reactor power [W]
        precursor_constants:
            ndarray, 1x6 array of lambda_i
        precursor_density:
            ndarray, 1x6 array of c_i

    Returns:
        ndarray 1x6 vector of the time derivative of each of the "i" components of precursor density

    """
    return beta_vector * power / period - precursor_constants * precursor_density


#################################################
#              THERMAL DYNAMICS                 #
#################################################


def mod_temp_deriv(heat_coeff: float, mass_mod: float, heat_cap_mod: float, mass_flow: float,
                   temp_fuel: float, temp_mod: float, temp_in: float) -> float:
    """Compute time derivative of moderator temperature, $\frac{dT_mod}{dt}(t)$

    Args:
        heat_coeff:
            float, heat transfer coefficient of fuel and moderator [J/K/sec]
        mass_mod:
            float, mass of moderator                               [kg]
        heat_cap_mod:
            float, specific Heat capacity of moderator             [J/kg/K]
        mass_flow:
            float, total moderator/coolant mass flow rate          [kg/sec]
        temp_fuel:
            float, temperature of fuel                             [K]
        temp_mod:
            float, temperature of moderator                        [K]
        temp_in:
            float, temperature of inlet coolant                    [K]
    """
    return (heat_coeff / (mass_mod * heat_cap_mod)) * (temp_fuel - temp_mod) - (2 * mass_flow / mass_mod) * (temp_mod - temp_in)


def fuel_temp_deriv(power: float, mass_fuel: float, heat_cap_fuel: float, heat_coeff: float, temp_fuel: float, temp_mod: float) -> float:
    """Compute time derivative of fuel temperature, $\frac{dT_fuel}{dt}(t)$

    Args:
        power:
            float, Reactor Power                                   [W]
        mass_fuel:
            float, mass of fuel                                    [kg]
        heat_cap_fuel:
            float, specific heat capacity of fuel                  [J/kg/K]
        heat_coeff:
            float, heat transfer coefficient of fuel and moderator [J/K/sec]
        temp_fuel:
            float, temperature of fuel                             [K]
        temp_mod:
            float, temperature of moderator                        [K]

    """
    return (power / (mass_fuel * heat_cap_fuel)) - ((heat_coeff / (mass_fuel * heat_cap_fuel)) * (temp_fuel - temp_mod))

#################################################
#                  REACTIVITY                   #
#################################################


def temp_fuel_reactivity_deriv(power:float, beta: float, mass_fuel: float, heat_cap_fuel: float, heat_coeff:float,
                               temp_fuel: float, temp_mod:float) -> float:
    """Compute time derivative of fuel temperature reactivity, $\frac{drho_fuel_temp}{dt}(t)$

    Args:
        beta:
            float, delayed neutron fraction                        []
        temp_fuel:
            float, temperature of fuel                             [K]

    """

    return beta * (((7.64e-7 * temp_fuel) - 3.36e-3) * fuel_temp_deriv(power=power, mass_fuel=mass_fuel, heat_cap_fuel=heat_cap_fuel,
                                                                      heat_coeff=heat_coeff, temp_fuel=temp_fuel, temp_mod=temp_mod))


def temp_mod_reactivity_deriv(beta: float, heat_coeff: float, mass_mod: float, heat_cap_mod: float, mass_flow: float,
                        temp_fuel:float, temp_mod: float, temp_in:float) -> float:
    """Compute time derivative of fuel temperature reactivity, $\frac{drho_mod_temp}{dt}(t)$

                Args:
                    beta:
                         float, delayed neutron fraction                            []
                    temp_mod:
                         float, temperature of moderator                            [K]

    """
    return beta * (((3.12e-7 * (temp_mod)) - 1.70e-3) * mod_temp_deriv(heat_coeff=heat_coeff, mass_mod=mass_mod, heat_cap_mod=heat_cap_mod,
                                                                      mass_flow=mass_flow, temp_fuel=temp_fuel, temp_mod=temp_mod, temp_in=temp_in))

def drum_angle_deriv(omega_drum: float) -> float:
    """
            Models rotation of drums, $\frac{ddrum_angle}{dt}(t)$

            Args:
                omega_drum:
                    float, rotation rate of control drums                       [degrees/sec]

    """
    return omega_drum



def con_drum_reactivity(beta: float, omega_drum:float, drum_angle: float) -> float:
    """Models control drum reactivity by the rotation of drums, $\frac{drho_control}{dt}(t)$

        Args:
            beta:
                float, delayed neutron fraction                          []
            drum_angle:
                float, angle of control drunk rotation                   [degrees]

    """
    return beta * ((1.953e-5 * ((drum_angle) ** 2) - (3.52e-3 * (drum_angle)) + 2.13e-2) * drum_angle_deriv(omega_drum=omega_drum))



