"""Solving Coupled ODEs

"""

import functools

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



class StateComponent:
    T_mod         = 7
    T_fuel        = 8
    rho_fuel_temp = 9
    theta_c       = 10


def total_neutron_deriv(beta: float, period: float, n, precursor_constants: np.ndarray,
                        precursor_density: np.ndarray, rho_fuel_temp: float, T_mod:float, fuel_gas_density:float,
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
                fuel_density_reactivity(beta, fuel_gas_density) + \
                modr_density_reactivity(beta, modr_gas_density) + \
                mods_density_reactivity(beta, mods_gas_density) + \
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

def fuel_temp_reactivity_deriv(beta: float, T_fuel: float) -> float:
    """Compute time derivative of fuel temperature reactivity, $\frac{drho_fuel_temp}{dt}(t)$

    Args:
        beta:
            float, delayed neutron fraction                        []
        T_fuel:
            float, temperature of fuel                             [K]

    """

    return beta * (7.64e-7 * T_fuel - 3.36e-3)

def theta_c_deriv(cdspd: float) -> float:
    """
            Models rotation of drums, $\frac{dtheta_c}{dt}(t)$

            Args:
                cdspd:
                    float, rotation rate of control drums                       [degrees/sec]
    """
    return cdspd


def _state_deriv(state: np.ndarray, t: float, beta_vector: np.ndarray, precursor_constants: np.ndarray,
                 total_beta: float, period: float, h: float, M_M: float, C_M: float, W_M: float, M_F: float, C_F: float,
                 T_in: float, fuel_gas_density:float, modr_gas_density:float, mods_gas_density:float, cdspd: float) -> np.ndarray:
    """Function to compute the time derivative of the reactor state, including the population count and the precursor densities

    Args:
        state:
            ndarray, 1x10 vector where the components represent the state of the reactor at time "t":
                - Component 0 is "n", total neutron population
                - Components 1-6 are "c_i", precursor densities
            These components are concatenated into a single array to conform to the scipy API
        t:
            float, currently unused parameter for scipy odeint interface

    Returns:
        ndarray, the time derivative of the reactor state at time "t"
    """
    dndt = total_neutron_deriv(beta=total_beta, period=period, n=state[0],
                               precursor_constants=precursor_constants, precursor_density=state[1:-4],
                               rho_fuel_temp= state[StateComponent.rho_fuel_temp], T_mod= state[StateComponent.T_mod],
                               fuel_gas_density=fuel_gas_density, modr_gas_density=modr_gas_density,
                               mods_gas_density=mods_gas_density, theta_c=state[StateComponent.theta_c])

    dcdt = delay_neutron_deriv(beta_vector=beta_vector, period=period, n=state[0],
                               precursor_constants=precursor_constants, precursor_density=state[1:-4])

    dT_moddt = mod_temp_deriv(h=h, M_M=M_M, C_M=C_M, W_M=W_M, T_fuel=state[StateComponent.T_fuel], T_mod=state[StateComponent.T_mod], T_in=T_in)

    dT_fueldt = fuel_temp_deriv(n=state[0], M_F=M_F, C_F=C_F, h=h, T_fuel=state[StateComponent.T_fuel], T_mod=state[StateComponent.T_mod])

    drho_fuel_temp_dt = fuel_temp_reactivity_deriv(beta=total_beta, T_fuel=[StateComponent.T_fuel])

    dtheta_c_dt = theta_c_deriv(cdspd=cdspd)

    state = np.concatenate((np.array([dndt]),
                            dcdt,
                            np.array([dT_moddt,
                                      dT_fueldt,
                                      drho_fuel_temp_dt,
                                      dtheta_c_dt])), axis=0)

    return state


class Solution:
    __slots__ = ('_array', '_t')

    def __init__(self, array: np.ndarray, t: np.ndarray):
        self._array = array
        self._t = t

    @property
    def array(self):
        return self._array

    @property
    def t(self):
        return self._t

    @property
    def neutron_population(self):
        return self._array[:, 0]

    @property
    def num_densities(self):
        return self.precursor_densities.shape[1]

    @property
    def precursor_densities(self):
        return self._array[:, 1:-4]

    def precursor_density(self, i: int):
        """Get a timeseries of precursor densitity of the ith kind

        Args:
            i:
                index of the vector component. These are 1-indexed, to match the mathematical notation.

        Returns:
            ndarray
        """
        return self.precursor_densities[:, i - 1]

    @property
    def T_mod(self):
        return self._array[:, 7]

    @property
    def T_fuel(self):
        return self._array[:, 8]

    @property
    def rho_fuel_temp(self):
        return self._array[:, 9]

    @property
    def theta_c(self):
        return self._array[:, 10]


def solve(n_initial: float, precursor_density_initial: np.ndarray, beta_vector: np.ndarray,
          precursor_constants: np.ndarray, total_beta: float, period: float, h: float,
          M_M: float, C_M: float, W_M: float, M_F: float, C_F: float, T_in: float, T_mod0: float, T_fuel0: float,
          rho_fuel_temp0: float, fuel_gas_density:float, modr_gas_density:float, cdspd: float, mods_gas_density: float, theta_c0: float,
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
    """

    initial_state = np.concatenate((np.array([n_initial]),
                                    precursor_density_initial,
                                    np.array([T_mod0,
                                              T_fuel0,
                                              rho_fuel_temp0,
                                              theta_c0])), axis=0)

    t = np.linspace(t_start, t_max, num_iters)

    deriv_func = functools.partial(_state_deriv,
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

    res = odeint(deriv_func, initial_state, t)

    return Solution(array=res, t=np.arange(t_start, t_max, (t_max - t_start) / num_iters))


def mod_temp_reactivity(beta: float, T_mod:float) -> float:
    """Compute reactivity due to moderator temperature.

                Args:
                    beta:
                         float, delayed neutron fraction                            []
                    T_mod:
                         float, temperature of moderator                            [K]

    """
    return beta * ((1.56e-7 * (T_mod) ** 2) - (1.70e-3 *(T_mod) + 0.666))

def fuel_density_reactivity(beta: float, fuel_gas_density: float) -> float:
    """Compute reactivity due to hydrogen gas density in the fuel.

                Args:
                    beta:
                         float, delayed neutron fraction                             []
                    fuel_gas_density:
                         float, gas density in the fuel                              [g/cc]

    """
    return beta * ((216.5 * fuel_gas_density) - 0.025)

def modr_density_reactivity(beta: float, modr_gas_density:float) -> float:
    """Compute reactivity due to hydrogen gas density in return channel of moderator.

                Args:
                    beta:
                        float, delayed neutron fraction                              []
                    modr_gas_density:
                        float, gas density in return channel of moderator            [g/cc]

        """

    return beta * ((97.6 * modr_gas_density) - 0.025)

def mods_density_reactivity(beta: float, mods_gas_density:float) -> float:
    """Compute reactivity due to hydrogen gas density in supply channel of moderator.

                Args:
                    beta:
                        float, delayed neutron fraction                              []
                    mods_gas_density:
                        float, gas density in supply channel of moderator            [g/cc]

        """

    return beta * ((32.2 * mods_gas_density) - 0.004)

def drum_reactivity(beta: float, theta_c: float) -> float:
    """Compute reactivity due to control drum rotation

        Args:
            beta:
                float, delayed neutron fraction                          []
            theta_c:
                float, angle of control drunk rotation                   [degrees]

    """
    return beta * ((6.51e-6 * (theta_c) ** 3) - (1.76e-3 * (theta_c) ** 2) + (2.13e-2 * theta_c) + 4.92536)

