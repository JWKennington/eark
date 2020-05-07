"""Module for numerical integrator utilities and interfaces. We define a "solve" function with EARK specific
arguments that relate to the state of the reactor and the initial conditions of the underlying dynamics.
"""

import functools

import numpy as np
from scipy.integrate import odeint

from eark import dynamics
from eark.solution import Solution
from eark.state import State


def state_deriv_array(state_array: np.ndarray, t: float, beta_vector: np.ndarray, precursor_constants: np.ndarray,
                      total_beta: float, period: float, h: float, M_M: float, C_M: float, W_M: float, M_F: float, C_F: float,
                      T_in: float, fuel_gas_density: float, modr_gas_density: float, mods_gas_density: float, cdspd: float) -> np.ndarray:
    """Function to compute the time derivative of the reactor state,

    Args:
        state_array:
            ndarray, 1x10 vector where the components represent the state of the reactor at time "t":
                - Component 0 is "n", total neutron population
                - Components 1-6 are "c_i", precursor densities
            These components are concatenated into a single array to conform to the scipy API
        t:
            float, currently unused parameter for scipy odeint interface

    Returns:
        ndarray, the time derivative of the reactor state at time "t"
    """
    state = State.from_array(state_array)

    dndt = dynamics.total_neutron_deriv(beta=total_beta, period=period, n=state.neutron_population,
                                        precursor_constants=precursor_constants, precursor_density=state.precursor_densities,
                                        rho_fuel_temp=state.rho_fuel_temp, T_mod=state.t_mod,
                                        fuel_gas_density=fuel_gas_density, modr_gas_density=modr_gas_density,
                                        mods_gas_density=mods_gas_density, theta_c=state.theta_c)

    dcdt = dynamics.delay_neutron_deriv(beta_vector=beta_vector, period=period, n=state.neutron_population,
                                        precursor_constants=precursor_constants, precursor_density=state.precursor_densities)

    dT_moddt = dynamics.mod_temp_deriv(h=h, M_M=M_M, C_M=C_M, W_M=W_M, T_fuel=state.t_fuel, T_mod=state.t_mod, T_in=T_in)

    dT_fueldt = dynamics.fuel_temp_deriv(n=state.neutron_population, M_F=M_F, C_F=C_F, h=h, T_fuel=state.t_fuel, T_mod=state.t_mod)

    drho_fuel_temp_dt = dynamics.fuel_temp_reactivity_deriv(beta=total_beta, T_fuel=state.t_fuel)

    dtheta_c_dt = dynamics.theta_c_deriv(cdspd=cdspd)

    state_deriv = State(dndt, dcdt, dT_moddt, dT_fueldt, drho_fuel_temp_dt, dtheta_c_dt)
    return state_deriv.to_array()


def solve(n_initial: float, precursor_density_initial: np.ndarray, beta_vector: np.ndarray,
          precursor_constants: np.ndarray, total_beta: float, period: float, h: float,
          M_M: float, C_M: float, W_M: float, M_F: float, C_F: float, T_in: float, T_mod0: float, T_fuel0: float,
          rho_fuel_temp0: float, fuel_gas_density: float, modr_gas_density: float, cdspd: float, mods_gas_density: float, theta_c0: float,
          t_max: float, t_start: float = 0, num_iters: int = 100) -> Solution:
    """Solving differential equations to calculate parameters of reactor at a certain state

    Args:
        n_initial:
            float, initial reactor power                                [W]
        precursor_density_initial:
            ndarray, 1x6 vector of initial precursor densities          []
        beta_vector:
            ndarray, 1x6 vector of beta_i                               []
        precursor_constants:
            ndarray, 1x6 vector of lambda_i                             []
        total_beta:
            float, delayed neutron fraction                             []
        period:
            float, effective generation time                            [sec]
        h:
            float, heat transfer coefficient of fuel and moderator      [J/K/sec]
        M_M:
            float, mass of moderator                                    [kg]
        C_M:
            float, specific Heat capacity of moderator                  [J/kg/K]
        W_M:
            float, total moderator/coolant mass flow rate               [kg/sec]
        M_F:
            float, mass of fuel                                         [kg]
        C_F:
            float, specific heat capacity of fuel                       [J/kg/K]
        T_fuel:
            float, temperature of fuel                                  [K]
        T_mod:
            float, temperature of moderator                             [K]
        T_in:
            float, temperature of inlet coolant                         [K]
        rho_fuel_temp0
            float, initial reactivity due to fuel temperature           [dk/K]
        cdspd:
            float, rotation rate of control drums                       [degrees/sec]
        theta_c:
            float, angle of control drunk rotation                      [degrees]
        fuel_gas_density:
            float, gas density in the fuel                              [g/cc]
        t_max:
            float, ending time of simulation                            [sec]
        t_start:
            float, default 0, starting time of simulation               [sec]
        num_iters:
            int, default 100, number of iterations                      []

    Returns:
        ndarray, state vector evolution 7xnum_iters

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    """
    # Build the initial state
    initial_state = State(n_initial, precursor_density_initial, T_mod0, T_fuel0, rho_fuel_temp0, theta_c0)

    # Compute time intervals for odeint integrator
    t = np.linspace(t_start, t_max, num_iters)

    # Partialize the state derivative function for signature compatibility with scipy.odeint, see [1] for "func" signature details
    deriv_func = functools.partial(state_deriv_array,
                                   beta_vector=beta_vector,
                                   precursor_constants=precursor_constants,
                                   total_beta=total_beta,
                                   period=period,
                                   h=h,
                                   M_M=M_M,
                                   C_M=C_M,
                                   W_M=W_M,
                                   M_F=M_F,
                                   C_F=C_F,
                                   T_in=T_in,
                                   fuel_gas_density=fuel_gas_density,
                                   modr_gas_density=modr_gas_density,
                                   mods_gas_density=mods_gas_density,
                                   cdspd=cdspd)

    # Compute result using odeint integrator, see [1] for numerical details
    res = odeint(deriv_func, initial_state.to_array(), t)

    # Create solution object
    return Solution(array=res, t=np.arange(t_start, t_max, (t_max - t_start) / num_iters))
