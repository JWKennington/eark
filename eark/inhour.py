"""Coupled differential equations representing neutron populations in reactor

[Vignesh can clean this up]

References:
    [1] Inhour Equation https://en.wikipedia.org/wiki/Inhour_equation
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

tmax = 10.0
L = 6.0e-05  # Prompt Generation Time
B = 0.0075  # TotalBeta


def total_neutron_deriv(rho: float, beta: float, period: float, n, precursor_constants: np.ndarray, precursor_density: np.ndarray) -> float:
    """Compute time derivative of total neutron population, $\frac{dn}{dt}(t)$

    Args:
        rho:
            float, the density
        beta:
            float, 
        period:
            float, reactor period
        n:
        precursor_constants:
        precursor_density:

    Returns:

    """
    return (((rho - beta) / period) * n) + np.inner(precursor_constants, precursor_density)

def delay_neutron_deriv(beta_vector: np.ndarray, period, n, precursor_constants: np.ndarray, precursor_density: np.ndarray) -> np.ndarray:
    return beta_vector * n / period - precursor_constants * precursor_density


def pop(s, t):
    bi = [0.033, 0.219, 0.196, 0.395, 0.115, 0.042]  # beta-i/beta
    li = [0.0124, 0.0305, 0.1110, 0.3011, 1.1400, 3.0100]  # lambda
    rho = 0.5 * B

    n = s[0]
    c1 = s[1]
    c2 = s[2]
    c3 = s[3]
    c4 = s[4]
    c5 = s[5]
    c6 = s[6]

    dndt = ((((rho - B) / L) * n) + (c1 * li[0]) + (c2 * li[1]) + (c3 * li[2]) + (c4 * li[3]) + (c5 * li[4]) + (c6 * li[5]))
    dc1dt = (((B * bi[0]) / L) * n) - (c1 * li[0])
    dc2dt = (((B * bi[1]) / L) * n) - (c1 * li[1])
    dc3dt = (((B * bi[2]) / L) * n) - (c1 * li[2])
    dc4dt = (((B * bi[3]) / L) * n) - (c1 * li[3])
    dc5dt = (((B * bi[4]) / L) * n) - (c1 * li[4])
    dc6dt = (((B * bi[5]) / L) * n) - (c1 * li[5])
    return [dndt, dc1dt, dc2dt, dc3dt, dc4dt, dc5dt, dc6dt]


s0 = [4000, 5000, 6000, 5600, 4700, 7800, 6578]
t = np.linspace(0, tmax, 100)

s = odeint(pop, s0, t)


