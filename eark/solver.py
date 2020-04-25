"""Solving Coupled ODEs to solve Power, Temperature and Reactivity for a given reactor state.

"""

import functools

import numpy as np
from scipy.integrate import odeint


def total_neutron_deriv(rho_temp: float, rho_con: float, beta: float, period: float, n, precursor_constants: np.ndarray,
                        precursor_density: np.ndarray) -> float:
    """Compute time derivative of total neutron population (i.e. reactor power), $\frac{dn}{dt}(t)$

    Args:
        rho:
            float, reactivity []
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

    Returns:
        float, the time derivative of total neutron population or reactor power

        Examples:
            Computing a sample time derivative
            >>> total_neutron_deriv(rho=0.5*0.0075,
            ...                     beta=0.0075,
            ...                     period=6.0e-05,
            ...                     n=4000,
            ...                     precursor_constants=np.array([0.0124, 0.0305, 0.1110, 0.3011, 1.1400, 3.0100]),
            ...                     precursor_density=np.array([5000, 6000, 5600, 4700, 7800, 6578]))
            -219026.44999999998
    """

    total_rho = rho_temp + rho_con
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

        Examples:
            Computing delayed neutron population vector derivative
            >>> delay_neutron_deriv(beta_vector=np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042]),
            ...                     period=0.0075,
            ...                     n=4000,
            ...                     precursor_constants=np.array([0.0124, 0.0305, 0.1110, 0.3011, 1.1400, 3.0100]),
            ...                     precursor_density=np.array([5000, 6000, 5600, 4700, 7800, 6578]))
            array([ 17538.        , 116617.        , 103911.73333333, 209251.49666667,
                    52441.33333333,   2600.22      ])
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
    """Compute time derivative of fuel temperature, $\frac{dT_mod}{dt}(t)$

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

def temp_reactivity_deriv(n:float, h: float, M_M: float, C_M: float, W_M: float, M_F: float, C_F: float, T_in: float,
                     T_mod: float, T_fuel: float, a_F: float, a_M: float) -> float:
    """Compute time derivative of reactivity due the temperature, $\frac{drho_temp}{dt}(t)$

        Args:
            n:
                float, Reactor Power                                   [W]
            M_F:
                float, mass of fuel                                    [kg]
            C_F:
                float, specific heat capacity of fuel                  [J/kg/K]
            M_M:
                float, mass of moderator                               [kg]
            C_M:
                float, specific Heat capacity of moderator             [J/kg/K]
            h:
                float, heat transfer coefficient of fuel and moderator [J/K/sec]
            W_M:
                float, total moderator/coolant mass flow rate          [kg/sec]
            a_F:
                float, temperature reactivity coefficient, fuel        [dK/K]
            a_M:
                float, temperature reactivity coefficient, moderator   [dK/K]
        """
    return (a_F * fuel_temp_deriv(n=n, M_F=M_F, C_F=C_F, h=h, T_fuel=T_fuel, T_mod=T_mod)) + \
           (a_M * mod_temp_deriv(h=h, M_M= M_M, C_M=C_M, W_M=W_M, T_fuel=T_fuel, T_mod=T_mod, T_in=T_in))

def theta_c_deriv(cdspd:float) -> float:
    """
            Models rotation of drums, $\frac{dtheta_c}{dt}(t)$

            Args:
                cdspd:
                    float, rotation rate of control drums                       [degrees/sec]
    """
    return cdspd

def con_reactivity_deriv(beta: float, cdspd:float, cdwrth: float, theta_c: float) -> float:
    """
        Models control drum reactivity by the rotation of drums, $\frac{drho_control}{dt}(t)$

        Args:
            beta:
                float, delayed neutron fraction                             []
            cdwrth:
                float, control worth for the full span of drum rotation     []
            theta_c:
                float, angle of control drunk rotation                      [degrees]
            cdspd:
                float, rotation rate of control drums                       [degrees/sec]

    """
    return beta * cdwrth * np.sin((theta_c * np.pi) / 180.) * theta_c_deriv(cdspd=cdspd)

def _state_deriv(state: np.ndarray, t: float, beta_vector: np.ndarray, precursor_constants: np.ndarray, total_beta: float,
                 period: float, h: float, M_M: float, C_M: float, W_M: float, M_F: float, C_F: float, T_in: float,
                 rho_con: float, a_F: float, a_M: float, cdspd:float, cdwrth: float) -> np.ndarray:
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
    dndt = total_neutron_deriv(rho_temp=state[9], rho_con=state[11], beta=total_beta, period=period, n=state[0],
                               precursor_constants=precursor_constants, precursor_density=state[1:-5])

    dcdt = delay_neutron_deriv(beta_vector=beta_vector, period=period, n=state[0],
                               precursor_constants=precursor_constants, precursor_density=state[1:-5])

    dT_moddt = mod_temp_deriv(h=h, M_M=M_M, C_M=C_M, W_M=W_M, T_fuel=state[8], T_mod=state[7], T_in=T_in)

    dT_fueldt = fuel_temp_deriv(n=state[0], M_F=M_F, C_F=C_F, h=h, T_fuel=state[8], T_mod=state[7])


    drho_temp_dt = temp_reactivity_deriv(n=state[0], h=h, M_M=M_M, C_M=C_M, W_M=W_M, M_F=M_F, C_F=C_F, T_in=T_in,
                                        T_mod=state[7], T_fuel=state[8], a_F=a_F, a_M=a_M)

    dtheta_c_dt = theta_c_deriv(cdspd=cdspd)

    drho_con_dt = con_reactivity_deriv(beta=total_beta, cdwrth=cdwrth, theta_c=state[10], cdspd=cdspd)

    state = np.concatenate((np.array([dndt]),
                            dcdt,
                            np.array([dT_moddt,
                                      dT_fueldt,
                                      drho_temp_dt,
                                      dtheta_c_dt,
                                      drho_con_dt])), axis=0)

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
        return self._array[:, 1:-5]

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
    def rho_temp(self):
        return self._array[:, 9]

    @property
    def theta_c(self):
        return self._array[:, 10]

    @property
    def rho_con(self):
        return self._array[:, 11]

def solve(n_initial: float,
          precursor_density_initial: np.ndarray, beta_vector: np.ndarray,
          precursor_constants: np.ndarray, total_beta: float, period: float, h: float, M_M: float,
          C_M: float, W_M: float, M_F: float, C_F: float, T_in: float, T_mod0: float, T_fuel0: float,
          rho_con: float, a_F: float, a_M: float, cdspd:float, cdwrth: float, theta_c0: float,
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
        rho_con:
             float, External Reactivity (i.e. control rods)             [dK]
         a_F:
            float, temperature reactivity coefficient, fuel             [dK/K]
         a_M:
            float, temperature reactivity coefficient, moderator        [dK/K]
        cdspd:
            float, rotation rate of control drums                       [degrees/sec]
        cdwrth:
            float, control worth for the full span of drum rotation     []
        theta_c:
            float, angle of control drunk rotation                      [degrees]
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
                                              0,
                                              theta_c0,
                                              rho_con])), axis=0)

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
                                   rho_con=rho_con,
                                   a_F=a_F,
                                   a_M=a_M,
                                   cdspd=cdspd,
                                   cdwrth=cdwrth)

    res = odeint(deriv_func, initial_state, t)

    return Solution(array=res, t=np.arange(t_start, t_max, (t_max - t_start) / num_iters))
